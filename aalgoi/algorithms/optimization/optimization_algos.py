"""
Optimization algorithms: knapsack variants.
All return {selected, value, weight, valid}.
"""

import random
import math
from typing import Any
from aalgoi.algorithms.base import Algorithm


class GreedyKnapsack(Algorithm):
    """Greedy 0/1 knapsack by value/weight ratio."""

    def __init__(self):
        super().__init__()
        self.name = "greedy_knapsack"
        self.time_complexity = "O(n log n)"
        self.space_complexity = "O(n)"
        self.tags = ["optimization", "knapsack", "greedy"]
        self.best_for = ["knapsack", "value_weight_ratio"]
        self.patterns = ["Greedy", "Knapsack"]
        self.problem_types = ["OPTIMIZATION"]

    def process(self, data: Any) -> dict:
        return self.solve(data)

    def validate_output(self, input_data: Any, output_data: Any) -> bool:
        if not isinstance(output_data, dict):
            return False
        return "selected" in output_data and "value" in output_data

    def solve(self, data: dict) -> dict:
        items = data.get("items", [])
        capacity = data.get("capacity", 0)
        if not items or capacity <= 0:
            return {"selected": [], "value": 0, "weight": 0, "valid": True}

        indexed = [(i, items[i]) for i in range(len(items))]
        indexed.sort(key=lambda x: x[1].get("value", x[1].get("val", 0)) /
                     max(x[1].get("weight", x[1].get("wt", 0)), 1), reverse=True)

        selected = []
        total_w = 0
        total_v = 0
        for i, item in indexed:
            w = item.get("weight", item.get("wt", 0))
            v = item.get("value", item.get("val", 0))
            if total_w + w <= capacity:
                selected.append(i)
                total_w += w
                total_v += v

        return {"selected": selected, "value": total_v, "weight": total_w, "valid": True}


class SimulatedAnnealing(Algorithm):
    """Simulated annealing for knapsack."""

    def __init__(self):
        super().__init__()
        self.name = "simulated_annealing"
        self.time_complexity = "O(n * iter)"
        self.space_complexity = "O(n)"
        self.tags = ["optimization", "knapsack", "metaheuristic", "combinatorial"]
        self.best_for = ["knapsack", "large_search_space"]
        self.patterns = ["Metaheuristic", "LocalSearch"]
        self.problem_types = ["OPTIMIZATION"]

    def process(self, data: Any) -> dict:
        return self.solve(data)

    def validate_output(self, input_data: Any, output_data: Any) -> bool:
        if not isinstance(output_data, dict):
            return False
        return "selected" in output_data and "value" in output_data

    def solve(self, data: dict) -> dict:
        items = data.get("items", [])
        capacity = data.get("capacity", 0)
        if not items or capacity <= 0:
            return {"selected": [], "value": 0, "weight": 0, "valid": True}

        n = len(items)
        current = [0] * n
        remaining = capacity
        indexed = sorted(range(n), key=lambda i: items[i].get("value", items[i].get("val", 0)) /
                         max(items[i].get("weight", items[i].get("wt", 0)), 1), reverse=True)
        for i in indexed:
            w = items[i].get("weight", items[i].get("wt", 0))
            if w <= remaining:
                current[i] = 1
                remaining -= w

        def fitness(chrom):
            tw = sum(items[i].get("weight", items[i].get("wt", 0)) for i in range(n) if chrom[i])
            if tw > capacity:
                return 0
            return sum(items[i].get("value", items[i].get("val", 0)) for i in range(n) if chrom[i])

        best = current[:]
        best_fit = fitness(best)
        temp = 100.0
        cooling = 0.95

        for _ in range(1000):
            neighbor = current[:]
            idx = random.randint(0, n - 1)
            neighbor[idx] = 1 - neighbor[idx]
            nf = fitness(neighbor)
            cf = fitness(current)
            delta = nf - cf
            if delta > 0 or (temp > 0 and random.random() < math.exp(delta / max(temp, 0.01))):
                current = neighbor
                if nf > best_fit:
                    best = neighbor[:]
                    best_fit = nf
            temp *= cooling

        selected = [i for i in range(n) if best[i]]
        total_v = sum(items[i].get("value", items[i].get("val", 0)) for i in selected)
        total_w = sum(items[i].get("weight", items[i].get("wt", 0)) for i in selected)
        return {"selected": selected, "value": total_v, "weight": total_w, "valid": total_w <= capacity}


class GeneticAlgorithm(Algorithm):
    """Genetic algorithm for knapsack."""

    def __init__(self):
        super().__init__()
        self.name = "genetic_algorithm"
        self.time_complexity = "O(pop * gen * n)"
        self.space_complexity = "O(pop * n)"
        self.tags = ["optimization", "knapsack", "evolutionary"]
        self.best_for = ["knapsack", "complex_constraints"]
        self.patterns = ["Evolutionary", "PopulationBased"]
        self.problem_types = ["OPTIMIZATION"]

    def process(self, data: Any) -> dict:
        return self.solve(data)

    def validate_output(self, input_data: Any, output_data: Any) -> bool:
        if not isinstance(output_data, dict):
            return False
        return "selected" in output_data and "value" in output_data

    def solve(self, data: dict) -> dict:
        items = data.get("items", [])
        capacity = data.get("capacity", 0)
        if not items or capacity <= 0:
            return {"selected": [], "value": 0, "weight": 0, "valid": True}

        n = len(items)
        pop_size = min(50, max(20, n * 5))
        generations = 100
        mutation_rate = 0.1

        def fitness(chrom):
            tw = sum(items[i].get("weight", items[i].get("wt", 0)) for i in range(n) if chrom[i])
            if tw > capacity:
                return 0
            return sum(items[i].get("value", items[i].get("val", 0)) for i in range(n) if chrom[i])

        population = [[1 if random.random() < 0.5 else 0 for _ in range(n)] for _ in range(pop_size)]
        best_ever = None
        best_fit = 0

        for _ in range(generations):
            fits = [fitness(c) for c in population]
            gbi = max(range(len(fits)), key=lambda i: fits[i])
            if fits[gbi] > best_fit:
                best_fit = fits[gbi]
                best_ever = population[gbi][:]

            new_pop = [best_ever[:]] if best_ever else []
            while len(new_pop) < pop_size:
                t1, t2 = random.sample(range(len(population)), min(2, len(population)))
                p1 = population[t1] if fits[t1] >= fits[t2] else population[t2]
                t3, t4 = random.sample(range(len(population)), min(2, len(population)))
                p2 = population[t3] if fits[t3] >= fits[t4] else population[t4]
                point = random.randint(1, max(1, n - 1))
                child = p1[:point] + p2[point:]
                for i in range(n):
                    if random.random() < mutation_rate:
                        child[i] = 1 - child[i]
                new_pop.append(child)
            population = new_pop

        if best_ever is None:
            best_ever = [0] * n

        selected = [i for i in range(n) if best_ever[i]]
        total_v = sum(items[i].get("value", items[i].get("val", 0)) for i in selected)
        total_w = sum(items[i].get("weight", items[i].get("wt", 0)) for i in selected)
        return {"selected": selected, "value": total_v, "weight": total_w, "valid": total_w <= capacity}


class HillClimbing(Algorithm):
    """Hill climbing for knapsack."""

    def __init__(self):
        super().__init__()
        self.name = "hill_climbing"
        self.time_complexity = "O(n * iter)"
        self.space_complexity = "O(n)"
        self.tags = ["optimization", "knapsack", "local_search"]
        self.best_for = ["knapsack", "local_optima"]
        self.patterns = ["LocalSearch", "IterativeImprovement"]
        self.problem_types = ["OPTIMIZATION"]

    def process(self, data: Any) -> dict:
        return self.solve(data)

    def validate_output(self, input_data: Any, output_data: Any) -> bool:
        if not isinstance(output_data, dict):
            return False
        return "selected" in output_data and "value" in output_data

    def solve(self, data: dict) -> dict:
        items = data.get("items", [])
        capacity = data.get("capacity", 0)
        if not items or capacity <= 0:
            return {"selected": [], "value": 0, "weight": 0, "valid": True}

        n = len(items)

        def fitness(chrom):
            tw = sum(items[i].get("weight", items[i].get("wt", 0)) for i in range(n) if chrom[i])
            if tw > capacity:
                return 0
            return sum(items[i].get("value", items[i].get("val", 0)) for i in range(n) if chrom[i])

        current = [0] * n
        sorted_idx = sorted(range(n), key=lambda i: items[i].get("value", items[i].get("val", 0)) /
                            max(items[i].get("weight", items[i].get("wt", 0)), 1), reverse=True)
        remaining = capacity
        for i in sorted_idx:
            w = items[i].get("weight", items[i].get("wt", 0))
            if w <= remaining:
                current[i] = 1
                remaining -= w

        best = current[:]
        best_fit = fitness(best)

        for _ in range(100):
            improved = False
            indices = list(range(n))
            random.shuffle(indices)
            for i in indices:
                neighbor = best[:]
                neighbor[i] = 1 - neighbor[i]
                nf = fitness(neighbor)
                if nf > best_fit:
                    best = neighbor
                    best_fit = nf
                    improved = True
            if not improved:
                break

        selected = [i for i in range(n) if best[i]]
        total_v = sum(items[i].get("value", items[i].get("val", 0)) for i in selected)
        total_w = sum(items[i].get("weight", items[i].get("wt", 0)) for i in selected)
        return {"selected": selected, "value": total_v, "weight": total_w, "valid": total_w <= capacity}


class ParticleSwarm(Algorithm):
    """Particle swarm optimization for knapsack."""

    def __init__(self):
        super().__init__()
        self.name = "particle_swarm"
        self.time_complexity = "O(particles * iter * n)"
        self.space_complexity = "O(particles * n)"
        self.tags = ["optimization", "knapsack", "swarm"]
        self.best_for = ["knapsack", "continuous_optimization"]
        self.patterns = ["SwarmIntelligence", "PopulationBased"]
        self.problem_types = ["OPTIMIZATION"]

    def process(self, data: Any) -> dict:
        return self.solve(data)

    def validate_output(self, input_data: Any, output_data: Any) -> bool:
        if not isinstance(output_data, dict):
            return False
        return "selected" in output_data and "value" in output_data

    def solve(self, data: dict) -> dict:
        items = data.get("items", [])
        capacity = data.get("capacity", 0)
        if not items or capacity <= 0:
            return {"selected": [], "value": 0, "weight": 0, "valid": True}

        n = len(items)
        num_particles = 30
        iterations = 100

        def fitness(chrom):
            tw = sum(items[i].get("weight", items[i].get("wt", 0)) for i in range(n) if chrom[i])
            if tw > capacity:
                return 0
            return sum(items[i].get("value", items[i].get("val", 0)) for i in range(n) if chrom[i])

        velocities = [[random.uniform(-1, 1) for _ in range(n)] for _ in range(num_particles)]
        positions = [[1 if random.random() < 0.5 else 0 for _ in range(n)] for _ in range(num_particles)]
        pbest = [p[:] for p in positions]
        pbest_fit = [fitness(p) for p in positions]

        gbest_idx = max(range(num_particles), key=lambda i: pbest_fit[i])
        gbest = pbest[gbest_idx][:]
        gbest_fit = pbest_fit[gbest_idx]

        w, c1, c2 = 0.7, 1.5, 1.5

        for _ in range(iterations):
            for i in range(num_particles):
                for j in range(n):
                    r1, r2 = random.random(), random.random()
                    velocities[i][j] = (w * velocities[i][j] +
                                        c1 * r1 * (pbest[i][j] - positions[i][j]) +
                                        c2 * r2 * (gbest[j] - positions[i][j]))
                    prob = 1.0 / (1.0 + math.exp(-velocities[i][j]))
                    positions[i][j] = 1 if random.random() < prob else 0

                fit = fitness(positions[i])
                if fit > pbest_fit[i]:
                    pbest[i] = positions[i][:]
                    pbest_fit[i] = fit
                    if fit > gbest_fit:
                        gbest = positions[i][:]
                        gbest_fit = fit

        selected = [i for i in range(n) if gbest[i]]
        total_v = sum(items[i].get("value", items[i].get("val", 0)) for i in selected)
        total_w = sum(items[i].get("weight", items[i].get("wt", 0)) for i in selected)
        return {"selected": selected, "value": total_v, "weight": total_w, "valid": total_w <= capacity}


class AntColony(Algorithm):
    """Ant colony optimization for knapsack."""

    def __init__(self):
        super().__init__()
        self.name = "ant_colony"
        self.time_complexity = "O(ants * iter * n)"
        self.space_complexity = "O(n)"
        self.tags = ["optimization", "knapsack", "swarm"]
        self.best_for = ["knapsack", "combinatorial_optimization"]
        self.patterns = ["SwarmIntelligence", "PheromoneBased"]
        self.problem_types = ["OPTIMIZATION"]

    def process(self, data: Any) -> dict:
        return self.solve(data)

    def validate_output(self, input_data: Any, output_data: Any) -> bool:
        if not isinstance(output_data, dict):
            return False
        return "selected" in output_data and "value" in output_data

    def solve(self, data: dict) -> dict:
        items = data.get("items", [])
        capacity = data.get("capacity", 0)
        if not items or capacity <= 0:
            return {"selected": [], "value": 0, "weight": 0, "valid": True}

        n = len(items)
        num_ants = 20
        iterations = 100
        pheromone = [1.0] * n
        alpha, beta = 1.0, 2.0
        evaporation = 0.5
        Q = 100.0

        def get_value(i):
            return items[i].get("value", items[i].get("val", 0))

        def get_weight(i):
            return items[i].get("weight", items[i].get("wt", 0))

        best_solution = None
        best_value = 0

        for _ in range(iterations):
            solutions = []
            for _ in range(num_ants):
                selected = []
                remaining = capacity
                available = list(range(n))
                random.shuffle(available)
                for i in available:
                    w = get_weight(i)
                    if w > remaining:
                        continue
                    v = get_value(i)
                    prob = (pheromone[i] ** alpha) * ((v / max(w, 1)) ** beta)
                    if random.random() < prob / (prob + 1):
                        selected.append(i)
                        remaining -= w

                total_v = sum(get_value(i) for i in selected)
                total_w = sum(get_weight(i) for i in selected)
                if total_w <= capacity and total_v > best_value:
                    best_value = total_v
                    best_solution = selected[:]
                solutions.append((selected, total_v))

            for i in range(n):
                pheromone[i] *= (1 - evaporation)
            for sol, val in solutions:
                for i in sol:
                    pheromone[i] += Q * val / max(sum(get_value(j) for j in sol), 1)

        if best_solution is None:
            best_solution = []

        total_v = sum(get_value(i) for i in best_solution)
        total_w = sum(get_weight(i) for i in best_solution)
        return {"selected": best_solution, "value": total_v, "weight": total_w, "valid": total_w <= capacity}
