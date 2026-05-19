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
