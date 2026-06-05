from dataclasses import dataclass
from typing import Any


@dataclass
class BenchmarkProblem:
    id: str
    domain: str
    difficulty: float
    problem_text: str
    data: Any
    expected_output: Any
    best_algorithm: str
    time_complexity: str
    verification_fn: str


def get_benchmark_problems() -> list[BenchmarkProblem]:
    return [
        BenchmarkProblem("sort_1", "integers", 0.1, "Sort the array of integers", {"nums": [3, 1, 4, 1, 5, 9, 2, 6]}, [1, 1, 2, 3, 4, 5, 6, 9], "tim_sort", "O(n log n)", "sorted"),
        BenchmarkProblem("sort_2", "integers", 0.1, "Sort the numbers in ascending order", {"nums": [5, 4, 3, 2, 1]}, [1, 2, 3, 4, 5], "tim_sort", "O(n log n)", "sorted"),
        BenchmarkProblem("sort_3", "integers", 0.15, "Sort the array, preserve order of equal elements", {"nums": [3, 1, 4, 1, 5]}, [1, 1, 3, 4, 5], "merge_sort", "O(n log n)", "sorted"),
        BenchmarkProblem("sort_4", "integers", 0.2, "Sort small array of at most 10 integers", {"nums": [7, 3, 5, 1, 9]}, [1, 3, 5, 7, 9], "insertion_sort", "O(n\xb2)", "sorted"),
        BenchmarkProblem("sort_5", "integers", 0.1, "Sort the integers using integer sort", {"nums": [100, 42, 7, 99, 3]}, [3, 7, 42, 99, 100], "radix_sort", "O(nk)", "sorted"),

        BenchmarkProblem("path_1", "graph", 0.3, "Find shortest path in weighted graph", {"edges": [[0,1,4],[0,2,2],[1,3,3],[2,1,1],[2,3,5]], "n": 4}, 4, "dijkstra", "O((V+E) log V)", "exact"),
        BenchmarkProblem("path_2", "graph", 0.2, "Find shortest path in unweighted graph from 0 to 3", {"edges": [[0,1],[0,2],[1,3],[2,3]], "n": 4}, 2, "bfs", "O(V+E)", "exact"),
        BenchmarkProblem("path_3", "graph", 0.4, "Find shortest path with negative weights", {"edges": [[0,1,1],[1,2,-2],[0,2,4]], "n": 3}, -1, "bellman_ford", "O(VE)", "exact"),
        BenchmarkProblem("path_4", "graph", 0.3, "Find all shortest paths from source", {"edges": [[0,1,1],[0,2,2],[1,3,1],[2,3,1]], "n": 4}, {0: 0, 1: 1, 2: 2, 3: 2}, "dijkstra", "O((V+E) log V)", "exact"),
        BenchmarkProblem("path_5", "graph", 0.3, "Check if path exists between two nodes", {"edges": [[0,1],[1,2],[3,4]], "n": 5, "start": 0, "end": 2}, True, "bfs", "O(V+E)", "exact"),

        BenchmarkProblem("search_1", "array", 0.15, "Find target in sorted array", {"nums": [1, 3, 5, 7, 9, 11, 13], "target": 7}, 3, "binary_search", "O(log n)", "exact"),
        BenchmarkProblem("search_2", "array", 0.15, "Find target in sorted array, return -1 if not found", {"nums": [2, 4, 6, 8, 10], "target": 5}, -1, "binary_search", "O(log n)", "exact"),
        BenchmarkProblem("search_3", "array", 0.2, "Find first occurrence of target", {"nums": [1, 2, 3, 3, 3, 4, 5], "target": 3}, 2, "binary_search", "O(log n)", "exact"),
        BenchmarkProblem("search_4", "array", 0.2, "Find insertion point for target in sorted array", {"nums": [1, 3, 5, 7], "target": 4}, 2, "binary_search", "O(log n)", "exact"),
        BenchmarkProblem("search_5", "array", 0.25, "Find minimum in rotated sorted array", {"nums": [4, 5, 6, 7, 0, 1, 2]}, 0, "binary_search", "O(log n)", "exact"),

        BenchmarkProblem("dp_1", "numbers", 0.4, "Find maximum sum subarray", {"nums": [-2, 1, -3, 4, -1, 2, 1, -5, 4]}, 6, "dp_running_max", "O(n)", "exact"),
        BenchmarkProblem("dp_2", "numbers", 0.5, "0/1 knapsack: maximize value within capacity", {"weights": [1, 3, 4, 5], "values": [1, 4, 5, 7], "capacity": 7}, 9, "knapsack", "O(n x capacity)", "exact"),
        BenchmarkProblem("dp_3", "numbers", 0.4, "Find minimum coins to make amount", {"coins": [1, 2, 5], "amount": 11}, 3, "coin_change", "O(n x amount)", "exact"),
        BenchmarkProblem("dp_4", "numbers", 0.45, "Find longest increasing subsequence length", {"nums": [10, 9, 2, 5, 3, 7, 101, 18]}, 4, "lis", "O(n log n)", "exact"),
        BenchmarkProblem("dp_5", "text", 0.5, "Find edit distance between two strings", {"s1": "horse", "s2": "ros"}, 3, "dp_2d_sequence", "O(n x m)", "exact"),

        BenchmarkProblem("greedy_1", "numbers", 0.3, "Select maximum non-overlapping intervals", {"intervals": [[1,3],[2,5],[3,9],[0,6],[5,7],[8,9]]}, 3, "activity_selection", "O(n log n)", "exact"),
        BenchmarkProblem("greedy_2", "numbers", 0.35, "Build optimal prefix code", {"freq": [5, 9, 12, 13, 16, 45]}, 224, "huffman", "O(n log n)", "approximate"),
        BenchmarkProblem("greedy_3", "graph", 0.4, "Find minimum spanning tree weight", {"edges": [[0,1,4],[0,2,3],[1,2,1],[1,3,2],[2,3,5]], "n": 4}, 6, "minimum_spanning_tree", "O(E log V)", "exact"),
        BenchmarkProblem("greedy_4", "numbers", 0.3, "Maximum profit from job scheduling", {"jobs": [[1,2,50],[3,5,20],[6,19,100],[2,100,200]]}, 250, "job_scheduling", "O(n log n)", "exact"),
        BenchmarkProblem("greedy_5", "numbers", 0.3, "Fractional knapsack maximize value", {"items": [[60,10],[100,20],[120,30]], "capacity": 50}, 240.0, "fractional_knapsack", "O(n log n)", "approximate"),

        BenchmarkProblem("bsearch_1", "numbers", 0.4, "Find minimum maximum after splitting array into m parts", {"nums": [7,2,5,10,8], "m": 2}, 18, "binary_search_max", "O(log(sum) x n)", "exact"),
        BenchmarkProblem("bsearch_2", "numbers", 0.4, "Find minimum eating speed to finish bananas in h hours", {"nums": [3,6,7,11], "h": 8}, 3, "binary_search_min", "O(log(max) x n)", "exact"),
        BenchmarkProblem("bsearch_3", "numbers", 0.35, "Find smallest divisor so sum is below threshold", {"nums": [1,2,5,9], "threshold": 6}, 5, "binary_search_min", "O(log(max) x n)", "exact"),
        BenchmarkProblem("bsearch_4", "numbers", 0.35, "Find kth smallest in sorted matrix", {"matrix": [[1,5,9],[10,11,13],[12,13,15]], "k": 8}, 13, "binary_search_min", "O(log(range) x n)", "exact"),
        BenchmarkProblem("bsearch_5", "numbers", 0.45, "Minimize maximum distance between gas stations", {"stations": [1,2,3,4,5,6,7,8,9,10], "k": 9}, 0.5, "binary_search_min", "O(log(range) x n)", "approximate"),

        BenchmarkProblem("hash_1", "integers", 0.2, "Find two numbers that sum to target", {"nums": [2,7,11,15], "target": 9}, [0, 1], "hash_complement", "O(n)", "set"),
        BenchmarkProblem("hash_2", "integers", 0.25, "Find elements appearing more than n/3 times", {"nums": [3,2,3]}, [3], "hash_frequency", "O(n)", "set"),
        BenchmarkProblem("hash_3", "integers", 0.3, "Count subarrays with sum equal to k", {"nums": [1,1,1], "k": 2}, 2, "dp_running_count", "O(n)", "exact"),
        BenchmarkProblem("hash_4", "text", 0.25, "Find first non-repeating character", {"s": "leetcode"}, 0, "hash_frequency", "O(n)", "exact"),
        BenchmarkProblem("hash_5", "text", 0.3, "Check if two strings are anagrams", {"s": "anagram", "t": "nagaram"}, True, "hash_frequency", "O(n)", "exact"),

        BenchmarkProblem("graph_1", "graph", 0.3, "Find topological order of DAG", {"edges": [[0,1],[0,2],[1,3],[2,3]], "n": 4}, [0, 1, 2, 3], "topological_sort", "O(V+E)", "set"),
        BenchmarkProblem("graph_2", "graph", 0.25, "Find number of connected components", {"edges": [[0,1],[1,2],[3,4]], "n": 5}, 2, "connected_components", "O(V+E)", "exact"),
        BenchmarkProblem("graph_3", "graph", 0.4, "Detect cycle in directed graph", {"edges": [[0,1],[1,2],[2,0]], "n": 3}, True, "dfs", "O(V+E)", "exact"),
        BenchmarkProblem("graph_4", "graph", 0.35, "Check if graph is bipartite", {"edges": [[0,1],[1,2],[2,3],[3,0]], "n": 4}, True, "bfs", "O(V+E)", "exact"),
        BenchmarkProblem("graph_5", "graph", 0.4, "Find number of islands in grid", {"grid": [[1,1,0,0,0],[1,1,0,0,0],[0,0,1,0,0],[0,0,0,1,1]]}, 3, "bfs", "O(V+E)", "exact"),

        BenchmarkProblem("string_1", "text", 0.3, "Find all occurrences of pattern in text", {"text": "abcabcabc", "pattern": "abc"}, [0, 3, 6], "rabin_karp", "O(n+m)", "exact"),
        BenchmarkProblem("string_2", "text", 0.35, "Compute edit distance between strings", {"s1": "intention", "s2": "execution"}, 5, "levenshtein", "O(n x m)", "exact"),
        BenchmarkProblem("string_3", "text", 0.3, "Find longest palindromic substring", {"s": "babad"}, "bab", "dp_2d_sequence", "O(n\xb2)", "exact"),
        BenchmarkProblem("string_4", "text", 0.25, "Check if string is valid parentheses", {"s": "()[]{}"}, True, "monotonic_stack", "O(n)", "exact"),
        BenchmarkProblem("string_5", "text", 0.3, "Find longest common prefix", {"strs": ["flower","flow","flight"]}, "fl", "divide_conquer", "O(n x m)", "exact"),

        BenchmarkProblem("ml_1", "feature_matrix", 0.5, "Classify data points", {"X": [[1,2],[2,3],[3,4],[5,6],[6,7]], "y": [0,0,0,1,1]}, None, "random_forest_classifier", "O(n x d x depth)", "approximate"),
        BenchmarkProblem("ml_2", "feature_matrix", 0.5, "Predict continuous values", {"X": [[1],[2],[3],[4],[5]], "y": [2,4,6,8,10]}, None, "linear_regression", "O(n x d\xb2)", "approximate"),
        BenchmarkProblem("ml_3", "feature_matrix", 0.5, "Group data into clusters", {"X": [[1,1],[1.5,1.5],[5,5],[5.5,5.5]], "k": 2}, None, "kmeans", "O(n x k x d x iter)", "approximate"),
        BenchmarkProblem("ml_4", "feature_matrix", 0.45, "Classify text documents", {"texts": ["good movie", "terrible film", "great acting", "bad plot"], "labels": [1,0,1,0]}, None, "text_classifier", "O(n x d)", "approximate"),
        BenchmarkProblem("ml_5", "feature_matrix", 0.55, "Classify with SVM", {"X": [[0,0],[1,1],[0,1],[1,0]], "y": [0,0,1,1]}, None, "svm_classifier", "O(n\xb2 x d)", "approximate"),

        BenchmarkProblem("math_1", "numbers", 0.15, "Compute GCD of two numbers", {"a": 48, "b": 18}, 6, "gcd", "O(log min(a,b))", "exact"),
        BenchmarkProblem("math_2", "numbers", 0.2, "Find all primes up to n", {"n": 30}, [2,3,5,7,11,13,17,19,23,29], "prime_sieve", "O(n log log n)", "exact"),
        BenchmarkProblem("math_3", "numbers", 0.2, "Compute modular exponentiation", {"base": 2, "exp": 10, "mod": 1000}, 24, "modular_arithmetic", "O(log exp)", "exact"),
        BenchmarkProblem("math_4", "numbers", 0.3, "Multiply two matrices", {"A": [[1,2],[3,4]], "B": [[5,6],[7,8]]}, [[19,22],[43,50]], "matrix_ops", "O(n\xb3)", "exact"),
        BenchmarkProblem("math_5", "numbers", 0.25, "Count number of 1 bits in integer", {"n": 11}, 3, "bit_manipulation", "O(log n)", "exact"),
    ]


def verify_solution(
    problem: BenchmarkProblem,
    actual: Any,
) -> bool:
    if actual is None:
        return False

    if problem.verification_fn == "exact":
        return actual == problem.expected_output

    elif problem.verification_fn == "sorted":
        if isinstance(actual, list) and isinstance(problem.expected_output, list):
            return sorted(actual) == sorted(problem.expected_output)
        return actual == problem.expected_output

    elif problem.verification_fn == "set":
        if isinstance(actual, list) and isinstance(problem.expected_output, list):
            def _normalize(lst):
                if lst and isinstance(lst[0], list):
                    return frozenset(tuple(x) for x in lst)
                return frozenset(lst)
            return _normalize(actual) == _normalize(problem.expected_output)
        return actual == problem.expected_output

    elif problem.verification_fn == "approximate":
        return actual is not None

    return actual == problem.expected_output


def run_benchmark(
    solve_fn,
    verbose: bool = False,
) -> dict:
    problems = get_benchmark_problems()
    results = {
        "total": len(problems),
        "correct": 0,
        "failed": 0,
        "errors": 0,
        "by_domain": {},
        "problems": [],
    }

    for problem in problems:
        try:
            actual = solve_fn(problem.problem_text, problem.data)
            correct = verify_solution(problem, actual)

            if correct:
                results["correct"] += 1
            else:
                results["failed"] += 1

            results["problems"].append({
                "id": problem.id,
                "domain": problem.domain,
                "correct": correct,
                "expected": problem.expected_output,
                "actual": actual,
            })

            if problem.domain not in results["by_domain"]:
                results["by_domain"][problem.domain] = {"correct": 0, "total": 0}
            results["by_domain"][problem.domain]["total"] += 1
            if correct:
                results["by_domain"][problem.domain]["correct"] += 1

        except Exception as e:
            results["errors"] += 1
            results["problems"].append({
                "id": problem.id,
                "domain": problem.domain,
                "correct": False,
                "error": str(e),
            })

            if problem.domain not in results["by_domain"]:
                results["by_domain"][problem.domain] = {"correct": 0, "total": 0}
            results["by_domain"][problem.domain]["total"] += 1

    results["accuracy"] = results["correct"] / max(results["total"], 1)
    return results
