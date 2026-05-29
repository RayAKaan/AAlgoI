import random
import math
from algorithms.base import Algorithm


class GreedyKnapsack(Algorithm):
    def __init__(self):
        super().__init__()
        self.name = "greedy_knapsack"
        self.time_complexity = "O(n log n)"
        self.space_complexity = "O(1)"
        self.tags = ["optimization", "knapsack", "greedy", "approximation"]
        self.best_for = ["resource_allocation", "fast_approximation"]
        self.patterns = ["Greedy", "Approximation"]
        self.problem_types = ["OPTIMIZATION"]

    def validate_output(self, input_data, output_data):
        return isinstance(output_data, dict) and "selected" in output_data and "value" in output_data

    def process(self, data):
        items = data['items']
        capacity = data['capacity']

        indexed_items = sorted(
            enumerate(items), 
            key=lambda x: x[1]['value']/x[1]['weight'], 
            reverse=True
        )

        total_weight = 0
        total_value = 0
        selected_indices = []

        for idx, item in indexed_items:
            if total_weight + item['weight'] <= capacity:
                selected_indices.append(idx)
                total_weight += item['weight']
                total_value += item['value']

        return {
            'selected': selected_indices,
            'value': total_value,
            'weight': total_weight
        }


class SimulatedAnnealing(Algorithm):
    def __init__(self):
        super().__init__()
        self.name = "simulated_annealing"
        self.time_complexity = "O(iterations)"
        self.space_complexity = "O(n)"
        self.tags = ["optimization", "metaheuristic", "combinatorial"]
        self.best_for = ["tsp", "knapsack", "complex_landscape"]
        self.patterns = ["Metaheuristic", "Combinatorial"]
        self.problem_types = ["OPTIMIZATION"]

    def validate_output(self, input_data, output_data):
        return isinstance(output_data, dict) and "selected" in output_data and "value" in output_data

    def process(self, data):
        items = data['items']
        capacity = data['capacity']
        iterations = data.get('iterations', 1000)

        def evaluate(state):
            val = sum(items[i]['value'] for i in range(len(items)) if state[i])
            wt = sum(items[i]['weight'] for i in range(len(items)) if state[i])
            return val if wt <= capacity else 0

        state = [random.choice([0, 1]) for _ in range(len(items))]
        best_state = state[:]
        best_val = evaluate(state)
        temp = 1000.0

        for i in range(iterations):
            neighbor = state[:]
            idx = random.randint(0, len(items)-1)
            neighbor[idx] = 1 - neighbor[idx]

            n_val = evaluate(neighbor)
            current_val = evaluate(state)

            if n_val > current_val or random.random() < math.exp((n_val - current_val) / temp):
                state = neighbor
                if n_val > best_val:
                    best_val = n_val
                    best_state = neighbor

            temp *= 0.995

        final_wt = sum(items[i]['weight'] for i in range(len(items)) if best_state[i])
        return {
            'selected': [i for i, x in enumerate(best_state) if x],
            'value': best_val,
            'weight': final_wt
        }


class GeneticAlgorithm(Algorithm):
    def __init__(self):
        super().__init__()
        self.name = "genetic_algorithm"
        self.time_complexity = "O(generations * population * fitness)"
        self.space_complexity = "O(population)"
        self.tags = ["optimization", "evolutionary", "genetic", "population"]
        self.best_for = ["combinatorial", "multi_modal", "large_search_space"]
        self.patterns = ["Evolutionary", "PopulationBased", "Crossover", "Mutation"]
        self.problem_types = ["OPTIMIZATION"]

    def validate_output(self, input_data, output_data):
        return isinstance(output_data, dict)

    def process(self, data):
        districts = data.get("districts", {})
        budget = data.get("budget", data.get("resources", 100))
        names = list(districts.keys())
        n = len(names)
        if n == 0:
            return {"selected": [], "value": 0}
        pop_size = min(50, 2 ** n)
        n_generations = 100
        mutation_rate = 0.1
        population = [[random.choice([0, 1]) for _ in range(n)] for _ in range(pop_size)]

        def fitness(individual):
            total = sum(districts[names[i]] for i in range(n) if individual[i])
            return total if sum(individual) <= budget else 0

        for gen in range(n_generations):
            scored = [(fitness(ind), ind) for ind in population]
            scored.sort(key=lambda x: -x[0])
            population = [ind for _, ind in scored[:pop_size // 2]]
            while len(population) < pop_size:
                p1 = random.choice(population)
                p2 = random.choice(population)
                split = random.randint(1, n - 1)
                child = p1[:split] + p2[split:]
                if random.random() < mutation_rate:
                    idx = random.randint(0, n - 1)
                    child[idx] = 1 - child[idx]
                population.append(child)

        best = max(population, key=fitness)
        selected = [i for i, v in enumerate(best) if v]
        return {"selected": selected, "value": fitness(best), "count": len(selected)}


class HillClimbing(Algorithm):
    def __init__(self):
        super().__init__()
        self.name = "hill_climbing"
        self.time_complexity = "O(iterations)"
        self.space_complexity = "O(n)"
        self.tags = ["optimization", "local_search", "iterative"]
        self.best_for = ["continuous_optimization", "local_optimum", "fast_convergence"]
        self.patterns = ["LocalSearch", "Iterative", "GradientFree"]
        self.problem_types = ["OPTIMIZATION"]

    def validate_output(self, input_data, output_data):
        return isinstance(output_data, dict)

    def process(self, data):
        districts = data.get("districts", {})
        resources = data.get("resources", data.get("budget", 5))
        names = list(districts.keys())
        n = len(names)
        if n == 0:
            return {"selected": [], "value": 0}

        def score(solution):
            return sum(districts[names[i]] for i in range(n) if solution[i])

        current = [0] * n
        for _ in range(min(resources, n)):
            candidates = [(i, districts[names[i]]) for i in range(n) if not current[i]]
            if not candidates:
                break
            candidates.sort(key=lambda x: -x[1])
            current[candidates[0][0]] = 1

        best_val = score(current)
        improved = True
        while improved:
            improved = False
            for i in range(n):
                if current[i]:
                    current[i] = 0
                    for j in range(n):
                        if not current[j]:
                            current[j] = 1
                            if score(current) > best_val:
                                best_val = score(current)
                                improved = True
                            else:
                                current[j] = 0
                    if not improved:
                        current[i] = 1

        selected = [i for i, v in enumerate(current) if v]
        return {"selected": selected, "value": best_val, "count": len(selected)}


class ParticleSwarmOptimization(Algorithm):
    def __init__(self):
        super().__init__()
        self.name = "pso"
        self.time_complexity = "O(iterations * particles)"
        self.space_complexity = "O(particles)"
        self.tags = ["optimization", "swarm", "particle_swarm", "continuous"]
        self.best_for = ["continuous_optimization", "multi_objective", "global_optimum"]
        self.patterns = ["SwarmIntelligence", "PopulationBased", "Continuous"]
        self.problem_types = ["OPTIMIZATION"]

    def validate_output(self, input_data, output_data):
        return isinstance(output_data, dict)

    def process(self, data):
        districts = data.get("districts", {})
        doses = data.get("doses", data.get("budget", data.get("resources", 1000)))
        names = list(districts.keys())
        n = len(names)
        if n == 0:
            return {"selected": [], "value": 0}
        n_particles = 30
        n_iterations = 100
        particles = [[random.random() for _ in range(n)] for _ in range(n_particles)]
        velocities = [[random.uniform(-1, 1) for _ in range(n)] for _ in range(n_particles)]
        pbest = [p[:] for p in particles]
        gbest = pbest[0][:]

        def evaluate(position):
            total = sum(districts[names[i]] for i in range(n) if position[i] > 0.5)
            return total if sum(1 for i in range(n) if position[i] > 0.5) <= doses / max(districts.values(), default=1) * n else 0

        for _ in range(n_iterations):
            for i in range(n_particles):
                if evaluate(particles[i]) > evaluate(pbest[i]):
                    pbest[i] = particles[i][:]
                if evaluate(pbest[i]) > evaluate(gbest):
                    gbest = pbest[i][:]
                for j in range(n):
                    r1, r2 = random.random(), random.random()
                    velocities[i][j] = (0.7 * velocities[i][j] + 1.5 * r1 * (pbest[i][j] - particles[i][j]) + 1.5 * r2 * (gbest[j] - particles[i][j]))
                    particles[i][j] += velocities[i][j]
                    particles[i][j] = max(0, min(1, particles[i][j]))

        selected = [i for i, v in enumerate(gbest) if v > 0.5]
        val = sum(districts[names[i]] for i in selected)
        return {"selected": selected, "value": val, "count": len(selected)}


class AntColonyOptimization(Algorithm):
    def __init__(self):
        super().__init__()
        self.name = "aco"
        self.time_complexity = "O(iterations * ants * nodes)"
        self.space_complexity = "O(n^2)"
        self.tags = ["optimization", "swarm", "ant_colony", "routing"]
        self.best_for = ["tsp", "vehicle_routing", "network_routing"]
        self.patterns = ["SwarmIntelligence", "PheromoneBased", "Combinatorial"]
        self.problem_types = ["OPTIMIZATION"]

    def validate_output(self, input_data, output_data):
        return isinstance(output_data, dict) and "path" in output_data

    def process(self, data):
        graph = data.get("graph", {})
        start = data.get("start", "")
        end = data.get("end", "")
        if not graph or not start or not end:
            return {"path": [], "cost": None}
        nodes = list(graph.keys())
        n_ants = 20
        n_iterations = 50
        evaporation = 0.5
        alpha, beta = 1, 2
        pheromone = {u: {v: 1.0 for v in graph[u]} for u in graph}
        best_path = None
        best_cost = float("inf")

        for _ in range(n_iterations):
            paths = []
            costs = []
            for _ in range(n_ants):
                path = [start]
                visited = {start}
                current = start
                cost = 0
                while current != end:
                    neighbors = [n for n in graph.get(current, {}) if n not in visited]
                    if not neighbors:
                        break
                    weights = []
                    for nbr in neighbors:
                        tau = pheromone[current].get(nbr, 1e-6)
                        eta = 1.0 / graph[current].get(nbr, 1)
                        weights.append(tau ** alpha * eta ** beta)
                    total = sum(weights)
                    probs = [w / total for w in weights]
                    r = random.random()
                    cum = 0
                    for i, nbr in enumerate(neighbors):
                        cum += probs[i]
                        if r <= cum:
                            path.append(nbr)
                            visited.add(nbr)
                            cost += graph[current].get(nbr, 0)
                            current = nbr
                            break
                if current == end:
                    paths.append(path)
                    costs.append(cost)
                    if cost < best_cost:
                        best_cost = cost
                        best_path = path[:]

            for u in pheromone:
                for v in pheromone[u]:
                    pheromone[u][v] *= (1 - evaporation)
            for path, cost in zip(paths, costs):
                if cost > 0:
                    deposit = 1.0 / cost
                    for i in range(len(path) - 1):
                        if path[i + 1] in pheromone.get(path[i], {}):
                            pheromone[path[i]][path[i + 1]] += deposit

        return {"path": best_path or [], "cost": best_cost if best_cost != float("inf") else None}
