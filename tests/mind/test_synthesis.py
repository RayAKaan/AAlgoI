import pytest
from aalgoi.core.mind.synthesis.templates import TemplateRegistry
from aalgoi.core.mind.synthesis.modifier import CodeModifier
from aalgoi.core.mind.synthesis.optimizer import CodeOptimizer
from aalgoi.core.mind.adapters.synthesizer_adapter import SynthesizerAdapter


@pytest.fixture
def templates():
    return TemplateRegistry()


@pytest.fixture
def modifier():
    return CodeModifier()


@pytest.fixture
def optimizer():
    return CodeOptimizer()


@pytest.fixture
def adapter():
    return SynthesizerAdapter()


class TestTemplateRegistry:
    def test_registry_has_templates(self, templates):
        assert len(templates._templates) > 0

    def test_all_templates_have_valid_code(self, templates):
        for t in templates._templates:
            code = t.generate({})
            assert code is not None
            try:
                compile(code, "<template>", "exec")
            except SyntaxError:
                pytest.fail(f"Template {t.name} generates invalid code")

    def test_dp_running_max_matches(self, templates):
        info = {
            "optimization_goal": "maximize",
            "metric": "sum",
            "contiguity": "contiguous",
            "domain": "integers",
        }
        code = templates.find_best("optimal_substructure", info)
        assert code is not None
        assert "def solve" in code
        assert "max" in code

    def test_dp_running_min_matches(self, templates):
        info = {
            "optimization_goal": "minimize",
            "metric": "sum",
            "contiguity": "contiguous",
            "domain": "integers",
        }
        code = templates.find_best("optimal_substructure", info)
        assert code is not None
        assert "min" in code

    def test_knapsack_matches(self, templates):
        info = {
            "optimization_goal": "maximize",
            "has_capacity": True,
        }
        code = templates.find_best("optimal_substructure", info)
        assert code is not None
        assert "capacity" in code

    def test_greedy_activity_matches(self, templates):
        info = {
            "optimization_goal": "maximize",
            "has_intervals": True,
        }
        code = templates.find_best("greedy_exchange", info)
        assert code is not None
        assert "sort" in code

    def test_binary_search_min_matches(self, templates):
        info = {
            "optimization_goal": "minimize",
            "has_search_space": True,
        }
        code = templates.find_best("monotonic_feasibility", info)
        assert code is not None
        assert "lo" in code and "hi" in code

    def test_hash_complement_matches(self, templates):
        info = {
            "optimization_goal": "find",
            "has_target": True,
            "has_pairs": True,
        }
        code = templates.find_best("hashing_fingerprint", info)
        assert code is not None
        assert "complement" in code

    def test_graph_bfs_matches(self, templates):
        info = {
            "domain": "graph",
            "optimization_goal": "find",
        }
        code = templates.find_best("graph_connectivity", info)
        assert code is not None
        assert "deque" in code or "queue" in code

    def test_brute_force_always_matches(self, templates):
        info = {}
        code = templates.find_best(None, info)
        assert code is not None
        assert "def solve" in code

    def test_no_match_returns_brute_force(self, templates):
        info = {"domain": "unknown", "optimization_goal": "unknown"}
        code = templates.find_best(None, info)
        assert code is not None

    def test_lis_matches_subsequence(self, templates):
        info = {
            "optimization_goal": "maximize",
            "metric": "length",
            "contiguity": "subsequence",
        }
        code = templates.find_best("optimal_substructure", info)
        assert code is not None
        assert "tails" in code

    def test_sliding_window_matches(self, templates):
        info = {
            "optimization_goal": "maximize",
            "metric": "length",
            "contiguity": "contiguous",
        }
        code = templates.find_best("amortized_invariant", info)
        assert code is not None

    def test_monotonic_stack_matches(self, templates):
        info = {
            "optimization_goal": "find",
            "needs_next_greater": True,
        }
        code = templates.find_best("amortized_invariant", info)
        assert code is not None
        assert "stack" in code


class TestCodeModifier:
    def test_add_memoization_recursive(self, modifier):
        code = '''def fib(n):
    if n <= 1:
        return n
    return fib(n-1) + fib(n-2)
'''
        result = modifier.modify(code, "add_memoization")
        assert result is not None
        assert "lru_cache" in result or "cache" in result

    def test_add_memoization_non_recursive(self, modifier):
        code = '''def solve(nums):
    return sum(nums)
'''
        result = modifier.modify(code, "add_memoization")
        assert result is None

    def test_sort_first(self, modifier):
        code = '''def solve(nums):
    return nums[0]
'''
        result = modifier.modify(code, "sort_first")
        assert result is not None
        assert "sort()" in result

    def test_sort_first_already_sorted(self, modifier):
        code = '''def solve(nums):
    nums.sort()
    return nums[0]
'''
        result = modifier.modify(code, "sort_first")
        assert result is None

    def test_unknown_modification(self, modifier):
        code = "def solve(x): return x"
        result = modifier.modify(code, "nonexistent_modification")
        assert result is None

    def test_add_hashing(self, modifier):
        code = '''def solve(nums, target):
    for i in range(len(nums)):
        for j in range(i+1, len(nums)):
            if nums[j] == target:
                return j
    return -1
'''
        result = modifier.modify(code, "add_hashing")
        assert result is None or "def solve" in result

    def test_modify_without_code(self, modifier):
        result = modifier.modify("", "add_memoization")
        assert result is None

    def test_modify_invalid_code(self, modifier):
        result = modifier.modify("not valid python {{{", "sort_first")
        assert result is None


class TestCodeOptimizer:
    def test_memoize_recursive(self, optimizer):
        code = '''def fib(n):
    if n <= 1:
        return n
    return fib(n-1) + fib(n-2)
'''
        result = optimizer.optimize(code, "memoize")
        assert result is not None
        assert "lru_cache" in result

    def test_memoize_non_recursive(self, optimizer):
        code = '''def solve(nums):
    return sum(nums)
'''
        result = optimizer.optimize(code, "memoize")
        assert result is None

    def test_two_pointer_nested_loops(self, optimizer):
        code = '''def solve(nums, target):
    for i in range(len(nums)):
        for j in range(i+1, len(nums)):
            if nums[i] + nums[j] == target:
                return [i, j]
    return []
'''
        result = optimizer.optimize(code, "two_pointer")
        assert result is not None
        assert "left" in result and "right" in result

    def test_two_pointer_no_nested_loops(self, optimizer):
        code = '''def solve(nums):
    return sum(nums)
'''
        result = optimizer.optimize(code, "two_pointer")
        assert result is None

    def test_rolling_array_dp(self, optimizer):
        code = '''def solve(nums):
    n = len(nums)
    dp = [0] * (n + 1)
    dp[0] = 1
    for i in range(1, n + 1):
        dp[i] = dp[i-1] + nums[i-1]
    return dp[n]
'''
        result = optimizer.optimize(code, "rolling_array")
        assert result is not None
        assert "prev" in result

    def test_bit_manipulation(self, optimizer):
        code = '''def solve(n):
    return n // 2 + n % 2
'''
        result = optimizer.optimize(code, "bit_manipulation")
        assert result is not None
        assert ">>" in result or "&" in result or "<<" in result

    def test_bit_manipulation_no_match(self, optimizer):
        code = '''def solve(n):
    return n + 1
'''
        result = optimizer.optimize(code, "bit_manipulation")
        assert result is None

    def test_unknown_optimization(self, optimizer):
        code = "def solve(x): return x"
        result = optimizer.optimize(code, "nonexistent")
        assert result is None

    def test_prefix_sum(self, optimizer):
        code = '''def solve(nums, queries):
    results = []
    for l, r in queries:
        results.append(sum(nums[l:r+1]))
    return results
'''
        result = optimizer.optimize(code, "prefix_sum")
        assert result is not None
        assert "prefix" in result


class TestSynthesizerAdapter:
    def test_synthesize_max_subarray(self, adapter):
        code = adapter.synthesize_novel(
            "Find the contiguous subarray with the largest sum",
            {"nums": [-2, 1, -3, 4, -1, 2, 1, -5, 4]},
            principle="optimal_substructure",
        )
        assert code is not None
        assert "def solve" in code
        assert "max" in code

    def test_synthesize_two_sum(self, adapter):
        code = adapter.synthesize_novel(
            "Find two numbers that add up to target",
            {"nums": [2, 7, 11, 15], "target": 9},
        )
        assert code is not None
        assert "def solve" in code

    def test_synthesize_shortest_path(self, adapter):
        code = adapter.synthesize_novel(
            "Find shortest path in graph from start to end",
            {"graph": {0: [1, 2], 1: [2], 2: [3]}},
        )
        assert code is not None

    def test_synthesize_unknown_problem(self, adapter):
        code = adapter.synthesize_novel(
            "Something completely unknown and weird",
            {},
        )
        assert code is not None

    def test_modify_adds_memoization(self, adapter):
        base_code = '''def fib(n):
    if n <= 1:
        return n
    return fib(n-1) + fib(n-2)
'''
        result = adapter.modify(
            base_code,
            "compute fibonacci",
            modification_type="add_memoization",
        )
        assert result is not None
        assert "lru_cache" in result or "cache" in result

    def test_modify_tries_all_types(self, adapter):
        base_code = '''def solve(nums):
    result = 0
    for num in nums:
        result += num
    return result
'''
        result = adapter.modify(base_code, "sum array")
        assert result is None or "def solve" in result

    def test_optimize_memoize(self, adapter):
        code = '''def fib(n):
    if n <= 1:
        return n
    return fib(n-1) + fib(n-2)
'''
        result = adapter.apply_optimization(code, "memoize")
        assert result is not None
        assert "lru_cache" in result

    def test_optimize_two_pointer(self, adapter):
        code = '''def solve(nums, target):
    for i in range(len(nums)):
        for j in range(i+1, len(nums)):
            if nums[i] + nums[j] == target:
                return [i, j]
    return []
'''
        result = adapter.apply_optimization(code, "two_pointer")
        assert result is not None

    def test_combine_algorithms(self, adapter):
        algo1 = {"algorithm": "sort", "code": "def solve(x): return sorted(x)"}
        algo2 = {"algorithm": "sum", "code": "def solve(x): return sum(x)"}
        result = adapter.combine(algo1, algo2)
        assert result is None or "def solve" in result

    def test_combine_empty_algorithms(self, adapter):
        result = adapter.combine({}, {})
        assert result is None

    def test_synthesize_with_no_data(self, adapter):
        code = adapter.synthesize_novel("sort an array", None)
        assert code is not None

    def test_extract_problem_info_target(self, adapter):
        info = adapter._extract_problem_info(
            "Find two numbers that sum to target", {"nums": [1, 2], "target": 3}
        )
        assert info["has_target"] is True
        assert info["has_pairs"] is True

    def test_extract_problem_info_graph(self, adapter):
        info = adapter._extract_problem_info(
            "Find shortest path in graph", {"edges": [[0, 1], [1, 2]]}
        )
        assert info["domain"] == "graph"

    def test_extract_problem_info_maximize(self, adapter):
        info = adapter._extract_problem_info(
            "Find the maximum sum subarray", None
        )
        assert info["optimization_goal"] == "maximize"

    def test_extract_problem_info_minimize(self, adapter):
        info = adapter._extract_problem_info(
            "Find the minimum cost path", None
        )
        assert info["optimization_goal"] == "minimize"

    def test_extract_problem_info_count(self, adapter):
        info = adapter._extract_problem_info(
            "Count the number of valid parentheses", None
        )
        assert info["optimization_goal"] == "count"


class TestTemplateExecution:
    def test_dp_running_max_executes(self, templates):
        info = {
            "optimization_goal": "maximize",
            "metric": "sum",
            "contiguity": "contiguous",
            "domain": "integers",
        }
        code = templates.find_best("optimal_substructure", info)
        assert code is not None

        namespace = {}
        exec(code, namespace)
        result = namespace["solve"]([-2, 1, -3, 4, -1, 2, 1, -5, 4])
        assert result == 6

    def test_hash_complement_executes(self, templates):
        info = {
            "optimization_goal": "find",
            "has_target": True,
            "has_pairs": True,
        }
        code = templates.find_best("hashing_fingerprint", info)
        assert code is not None

        namespace = {}
        exec(code, namespace)
        result = namespace["solve"]([2, 7, 11, 15], 9)
        assert result == [0, 1]

    def test_knapsack_executes(self, templates):
        info = {
            "optimization_goal": "maximize",
            "has_capacity": True,
        }
        code = templates.find_best("optimal_substructure", info)
        assert code is not None

        namespace = {}
        exec(code, namespace)
        result = namespace["solve"]([1, 3, 4, 5], [1, 4, 5, 7], 7)
        assert result == 9

    def test_graph_bfs_executes(self, templates):
        info = {
            "domain": "graph",
            "optimization_goal": "find",
        }
        code = templates.find_best("graph_connectivity", info)
        assert code is not None

        namespace = {}
        exec(code, namespace)
        graph = {0: [1, 2], 1: [2, 3], 2: [3], 3: []}
        result = namespace["solve"](graph, 0, 3)
        assert result == 2

    def test_greedy_activity_executes(self, templates):
        info = {
            "optimization_goal": "maximize",
            "has_intervals": True,
        }
        code = templates.find_best("greedy_exchange", info)
        assert code is not None

        namespace = {}
        exec(code, namespace)
        intervals = [[1, 3], [2, 5], [3, 9], [0, 6], [5, 7], [8, 9]]
        result = namespace["solve"](intervals)
        assert result == 3

    def test_lis_executes(self, templates):
        info = {
            "optimization_goal": "maximize",
            "metric": "length",
            "contiguity": "subsequence",
        }
        code = templates.find_best("optimal_substructure", info)
        assert code is not None

        namespace = {}
        exec(code, namespace)
        result = namespace["solve"]([10, 9, 2, 5, 3, 7, 101, 18])
        assert result == 4

    def test_binary_search_min_executes(self, templates):
        info = {
            "optimization_goal": "minimize",
            "has_search_space": True,
        }
        code = templates.find_best("monotonic_feasibility", info)
        assert code is not None

        namespace = {}
        exec(code, namespace)
        result = namespace["solve"]([3, 6, 7, 11], 8)
        assert result == 4

    def test_monotonic_stack_executes(self, templates):
        info = {
            "optimization_goal": "find",
            "needs_next_greater": True,
        }
        code = templates.find_best("amortized_invariant", info)
        assert code is not None

        namespace = {}
        exec(code, namespace)
        result = namespace["solve"]([4, 5, 2, 25])
        assert result == [5, 25, 25, -1]
