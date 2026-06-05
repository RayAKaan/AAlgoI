
import copy
import random
from typing import Any


class GeneticPipelineEvolver:
    def __init__(self, algo_pool: dict[str, Any], pop_size: int = 20,
                 mutation_rate: float = 0.2, crossover_rate: float = 0.7):
        self.pool = list(algo_pool.keys())
        self.pool_objects = algo_pool
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.population: list[list[str]] = self._random_population()
        self.fitness_history: list[float] = []
        self.generation = 0
        self.best_individual: list[str] | None = None
        self.best_fitness: float = 0.0

    def _random_population(self) -> list[list[str]]:
        return [
            random.sample(self.pool, k=random.randint(1, min(4, len(self.pool))))
            for _ in range(self.pop_size)
        ]

    def evaluate_fitness(self, performance_data: dict[str, dict]) -> list[float]:
        fitness_scores = []

        for individual in self.population:
            if not individual:
                fitness_scores.append(0.0)
                continue

            scores = []
            for algo_name in individual:
                if algo_name in performance_data:
                    perf = performance_data[algo_name]
                    avg_score = perf.get("avg_score", 0.0)
                    count = perf.get("count", 0)
                    confidence = min(1.0, count / 50.0)
                    scores.append(avg_score * confidence)
                else:
                    scores.append(0.0)

            pipeline_fitness = sum(scores) / len(scores) if scores else 0.0

            length_bonus = 1.0 / len(individual)
            pipeline_fitness = 0.7 * pipeline_fitness + 0.3 * length_bonus

            fitness_scores.append(pipeline_fitness)

        return fitness_scores

    def evolve(self, fitness_scores: list[float]):
        ranked = sorted(zip(fitness_scores, self.population), key=lambda x: x[0], reverse=True)
        parents = [p for _, p in ranked[:self.pop_size // 2]]

        if ranked and ranked[0][0] > self.best_fitness:
            self.best_fitness = ranked[0][0]
            self.best_individual = ranked[0][1]

        children = []
        for i in range(0, len(parents) - 1, 2):
            if i + 1 < len(parents) and random.random() < self.crossover_rate:
                child = self._crossover(parents[i], parents[i + 1])
            else:
                child = copy.deepcopy(parents[i])
            child = self._mutate(child)
            children.append(child)

        while len(children) + len(parents) < self.pop_size:
            children.append(self._mutate(random.sample(self.pool, k=random.randint(1, 3))))

        self.population = parents + children[:self.pop_size - len(parents)]
        self.generation += 1

        if fitness_scores:
            self.fitness_history.append(sum(fitness_scores) / len(fitness_scores))

    def _crossover(self, parent_a: list[str], parent_b: list[str]) -> list[str]:
        split = len(parent_a) // 2
        child = parent_a[:split] + parent_b[split:]
        seen = set()
        unique = []
        for algo in child:
            if algo not in seen:
                seen.add(algo)
                unique.append(algo)
        return unique if unique else parent_a[:1]

    def _mutate(self, individual: list[str]) -> list[str]:
        if not individual:
            return random.sample(self.pool, k=1)

        mutated = copy.deepcopy(individual)
        if random.random() < self.mutation_rate:
            idx = random.randint(0, len(mutated) - 1)
            alternatives = [a for a in self.pool if a != mutated[idx]]
            if alternatives:
                mutated[idx] = random.choice(alternatives)

        if random.random() < self.mutation_rate * 0.5:
            new_algo = random.choice(self.pool)
            if new_algo not in mutated:
                mutated.append(new_algo)

        if len(mutated) > 1 and random.random() < self.mutation_rate * 0.3:
            mutated.pop(random.randint(0, len(mutated) - 1))

        if not mutated:
            mutated = random.sample(self.pool, k=1)

        return mutated

    def get_best(self) -> list[str] | None:
        return self.best_individual

    def get_best_pipeline(self) -> list[Any]:
        if not self.best_individual:
            return []
        return [self.pool_objects[name] for name in self.best_individual if name in self.pool_objects]

    def get_stats(self) -> dict[str, Any]:
        return {
            "generation": self.generation,
            "population_size": len(self.population),
            "best_fitness": self.best_fitness,
            "best_individual": self.best_individual,
            "fitness_history": self.fitness_history[-10:] if self.fitness_history else []
        }
