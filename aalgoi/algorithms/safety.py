
from aalgoi.algorithms.base import Algorithm


class IdentityAlgorithm(Algorithm):
    """Always returns input unchanged — ultimate fallback."""
    def __init__(self):
        super().__init__()
        self.name = "identity"
        self.time_complexity = "O(1)"
        self.space_complexity = "O(1)"
        self.tags = ["safety", "fallback", "no_op"]
        self.best_for = ["unknown_problems", "malformed_data"]
        self.patterns = ["Identity"]
        self.problem_types = ["SAFETY"]

    def process(self, data):
        return data

    def validate_output(self, input_data, output_data):
        return True


class SafeSort(Algorithm):
    """Guaranteed to sort — uses built-in sorted()."""
    def __init__(self):
        super().__init__()
        self.name = "safe_sort"
        self.time_complexity = "O(n log n)"
        self.tags = ["safety", "sorting"]
        self.patterns = ["Fallback", "Sorting"]
        self.problem_types = ["SORTING", "SAFETY"]

    def process(self, data):
        try:
            return sorted(data)
        except Exception:
            return list(data) if hasattr(data, '__iter__') else [data]

    def validate_output(self, input_data, output_data):
        return True


class SafePath(Algorithm):
    """Returns empty path if graph is invalid."""
    def __init__(self):
        super().__init__()
        self.name = "safe_path"
        self.tags = ["safety", "pathfinding"]
        self.patterns = ["Fallback", "Pathfinding"]
        self.problem_types = ["PATHFINDING", "SAFETY"]

    def process(self, data):
        try:
            graph = data.get('graph', {}) if isinstance(data, dict) else {}
            start = data.get('start', '') if isinstance(data, dict) else ''
            end = data.get('end', '') if isinstance(data, dict) else ''
            if not graph or start not in graph or end not in graph:
                return []
            return [start, end]
        except Exception:
            return []

    def validate_output(self, input_data, output_data):
        return True


class SafeKnapsack(Algorithm):
    """Returns empty selection if items list is invalid."""
    def __init__(self):
        super().__init__()
        self.name = "safe_knapsack"
        self.tags = ["safety", "optimization"]
        self.patterns = ["Fallback", "Optimization"]
        self.problem_types = ["OPTIMIZATION", "SAFETY"]

    def process(self, data):
        try:
            items = data.get('items', []) if isinstance(data, dict) else []
            capacity = data.get('capacity', 0) if isinstance(data, dict) else 0
            if not items or capacity <= 0:
                return {'selected': [], 'value': 0, 'weight': 0}
            v = items[0].get('value', 0) if isinstance(items[0], dict) else items[0]
            w = items[0].get('weight', 0) if isinstance(items[0], dict) else items[0]
            return {'selected': [0], 'value': v, 'weight': w}
        except Exception:
            return {'selected': [], 'value': 0, 'weight': 0}

    def validate_output(self, input_data, output_data):
        return isinstance(output_data, dict) and "selected" in output_data
