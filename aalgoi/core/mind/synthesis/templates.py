from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class CodeTemplate:
    name: str
    principle: str
    description: str
    applicability: Callable[[dict], bool]
    generate: Callable[[dict], str]


class TemplateRegistry:
    def __init__(self) -> None:
        self._templates: list[CodeTemplate] = []
        self._register_all()

    def find_best(
        self,
        principle: str | None,
        problem_info: dict,
    ) -> str | None:
        candidates = []
        for t in self._templates:
            if principle and t.principle != principle:
                continue
            if t.applicability(problem_info):
                candidates.append(t)

        for t in candidates:
            code = t.generate(problem_info)
            if code and self._syntax_valid(code):
                return code

        if principle:
            for t in self._templates:
                if t.applicability(problem_info):
                    code = t.generate(problem_info)
                    if code and self._syntax_valid(code):
                        return code

        return None

    def _syntax_valid(self, code: str) -> bool:
        try:
            compile(code, "<template>", "exec")
            return True
        except SyntaxError:
            return False

    def _register_all(self) -> None:
        self._templates = [
            self._dp_running_max(),
            self._dp_running_min(),
            self._dp_running_count(),
            self._dp_knapsack_01(),
            self._dp_knapsack_unbounded(),
            self._dp_lis(),
            self._dp_2d_sequence(),
            self._greedy_activity_selection(),
            self._greedy_sort_scan(),
            self._binary_search_min(),
            self._binary_search_max(),
            self._two_pointer_opposite(),
            self._sliding_window(),
            self._hash_complement(),
            self._hash_frequency(),
            self._monotonic_stack(),
            self._graph_bfs(),
            self._graph_dfs(),
            self._prefix_sum(),
            self._brute_force(),
        ]

    def _dp_running_max(self) -> CodeTemplate:
        def applicable(info: dict) -> bool:
            return (
                info.get("optimization_goal") in ("maximize", "find")
                and info.get("metric") in ("sum", "product", "value")
                and info.get("contiguity") != "subsequence"
                and info.get("domain") in ("integers", "numbers", "array")
            )

        def generate(info: dict) -> str:
            return '''def solve(nums):
    if not nums:
        return 0
    best = curr = nums[0]
    for i in range(1, len(nums)):
        curr = max(nums[i], curr + nums[i])
        best = max(best, curr)
    return best
'''

        return CodeTemplate(
            name="dp_running_max",
            principle="optimal_substructure",
            description="Kadane-style: max running computation on array",
            applicability=applicable,
            generate=generate,
        )

    def _dp_running_min(self) -> CodeTemplate:
        def applicable(info: dict) -> bool:
            return (
                info.get("optimization_goal") == "minimize"
                and info.get("metric") in ("sum", "cost", "value")
                and info.get("contiguity") != "subsequence"
                and info.get("domain") in ("integers", "numbers", "array")
            )

        def generate(info: dict) -> str:
            return '''def solve(nums):
    if not nums:
        return 0
    best = curr = nums[0]
    for i in range(1, len(nums)):
        curr = min(nums[i], curr + nums[i])
        best = min(best, curr)
    return best
'''

        return CodeTemplate(
            name="dp_running_min",
            principle="optimal_substructure",
            description="Kadane-style: min running computation on array",
            applicability=applicable,
            generate=generate,
        )

    def _dp_running_count(self) -> CodeTemplate:
        def applicable(info: dict) -> bool:
            return (
                info.get("optimization_goal") == "count"
                and info.get("domain") in ("integers", "numbers", "array")
            )

        def generate(info: dict) -> str:
            return '''def solve(nums):
    if not nums:
        return 0
    count = 0
    prefix_sum = 0
    seen = {0: 1}
    for num in nums:
        prefix_sum += num
        count += seen.get(prefix_sum, 0)
        seen[prefix_sum] = seen.get(prefix_sum, 0) + 1
    return count
'''

        return CodeTemplate(
            name="dp_running_count",
            principle="optimal_substructure",
            description="Count subarrays with property using prefix sums",
            applicability=applicable,
            generate=generate,
        )

    def _dp_knapsack_01(self) -> CodeTemplate:
        def applicable(info: dict) -> bool:
            return (
                info.get("optimization_goal") in ("maximize", "minimize")
                and info.get("has_capacity") is True
                and info.get("items_are_discrete") is not False
            )

        def generate(info: dict) -> str:
            return '''def solve(weights, values, capacity):
    n = len(weights)
    dp = [0] * (capacity + 1)
    for i in range(n):
        for w in range(capacity, weights[i] - 1, -1):
            dp[w] = max(dp[w], dp[w - weights[i]] + values[i])
    return dp[capacity]
'''

        return CodeTemplate(
            name="dp_knapsack_01",
            principle="optimal_substructure",
            description="0/1 knapsack: space-optimized DP",
            applicability=applicable,
            generate=generate,
        )

    def _dp_knapsack_unbounded(self) -> CodeTemplate:
        def applicable(info: dict) -> bool:
            return (
                info.get("optimization_goal") in ("maximize", "minimize", "count")
                and info.get("has_capacity") is True
                and info.get("items_reusable") is True
            )

        def generate(info: dict) -> str:
            return '''def solve(coins, amount):
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    for i in range(1, amount + 1):
        for c in coins:
            if c <= i:
                dp[i] = min(dp[i], dp[i - c] + 1)
    return dp[amount] if dp[amount] != float('inf') else -1
'''

        return CodeTemplate(
            name="dp_knapsack_unbounded",
            principle="optimal_substructure",
            description="Unbounded knapsack / coin change",
            applicability=applicable,
            generate=generate,
        )

    def _dp_lis(self) -> CodeTemplate:
        def applicable(info: dict) -> bool:
            return (
                info.get("optimization_goal") in ("maximize", "find")
                and info.get("metric") in ("length", "count")
                and info.get("contiguity") == "subsequence"
            )

        def generate(info: dict) -> str:
            return '''def solve(nums):
    if not nums:
        return 0
    tails = []
    for num in nums:
        lo, hi = 0, len(tails)
        while lo < hi:
            mid = (lo + hi) // 2
            if tails[mid] < num:
                lo = mid + 1
            else:
                hi = mid
        if lo == len(tails):
            tails.append(num)
        else:
            tails[lo] = num
    return len(tails)
'''

        return CodeTemplate(
            name="dp_lis",
            principle="optimal_substructure",
            description="Longest increasing subsequence (patience sorting)",
            applicability=applicable,
            generate=generate,
        )

    def _dp_2d_sequence(self) -> CodeTemplate:
        def applicable(info: dict) -> bool:
            return (
                info.get("optimization_goal") in ("minimize", "maximize", "find")
                and info.get("has_two_sequences") is True
            )

        def generate(info: dict) -> str:
            return '''def solve(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        dp[i][0] = i
    for j in range(1, n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return dp[m][n]
'''

        return CodeTemplate(
            name="dp_2d_sequence",
            principle="optimal_substructure",
            description="2D sequence DP (edit distance / LCS)",
            applicability=applicable,
            generate=generate,
        )

    def _greedy_activity_selection(self) -> CodeTemplate:
        def applicable(info: dict) -> bool:
            return (
                info.get("optimization_goal") in ("maximize", "count")
                and info.get("has_intervals") is True
            )

        def generate(info: dict) -> str:
            return '''def solve(intervals):
    if not intervals:
        return 0
    intervals.sort(key=lambda x: x[1])
    count = 1
    end = intervals[0][1]
    for i in range(1, len(intervals)):
        if intervals[i][0] >= end:
            count += 1
            end = intervals[i][1]
    return count
'''

        return CodeTemplate(
            name="greedy_activity_selection",
            principle="greedy_exchange",
            description="Activity selection: sort by end, greedy pick",
            applicability=applicable,
            generate=generate,
        )

    def _greedy_sort_scan(self) -> CodeTemplate:
        def applicable(info: dict) -> bool:
            return (
                info.get("optimization_goal") in ("maximize", "minimize", "find")
                and info.get("domain") in ("integers", "numbers", "array")
                and info.get("can_sort") is not False
            )

        def generate(info: dict) -> str:
            return '''def solve(nums):
    nums.sort()
    result = 0
    for i, num in enumerate(nums):
        result = max(result, num * (len(nums) - i))
    return result
'''

        return CodeTemplate(
            name="greedy_sort_scan",
            principle="greedy_exchange",
            description="Sort then scan for optimal",
            applicability=applicable,
            generate=generate,
        )

    def _binary_search_min(self) -> CodeTemplate:
        def applicable(info: dict) -> bool:
            return (
                info.get("optimization_goal") == "minimize"
                and info.get("has_search_space") is True
            )

        def generate(info: dict) -> str:
            return '''def solve(nums, threshold):
    lo, hi = 1, max(nums)
    while lo < hi:
        mid = (lo + hi) // 2
        total = sum((x + mid - 1) // mid for x in nums)
        if total <= threshold:
            hi = mid
        else:
            lo = mid + 1
    return lo
'''

        return CodeTemplate(
            name="binary_search_min",
            principle="monotonic_feasibility",
            description="Binary search for minimum feasible value",
            applicability=applicable,
            generate=generate,
        )

    def _binary_search_max(self) -> CodeTemplate:
        def applicable(info: dict) -> bool:
            return (
                info.get("optimization_goal") == "maximize"
                and info.get("has_search_space") is True
            )

        def generate(info: dict) -> str:
            return '''def solve(nums, m):
    lo, hi = max(nums), sum(nums)
    while lo < hi:
        mid = (lo + hi) // 2
        count = 1
        curr = 0
        for x in nums:
            if curr + x > mid:
                count += 1
                curr = 0
            curr += x
        if count <= m:
            hi = mid
        else:
            lo = mid + 1
    return lo
'''

        return CodeTemplate(
            name="binary_search_max",
            principle="monotonic_feasibility",
            description="Binary search for maximum feasible value",
            applicability=applicable,
            generate=generate,
        )

    def _two_pointer_opposite(self) -> CodeTemplate:
        def applicable(info: dict) -> bool:
            return (
                info.get("optimization_goal") in ("maximize", "minimize", "find")
                and info.get("has_pairs") is True
                and info.get("domain") in ("integers", "numbers", "array")
            )

        def generate(info: dict) -> str:
            return '''def solve(nums, target):
    nums.sort()
    left, right = 0, len(nums) - 1
    best = float('inf')
    while left < right:
        s = nums[left] + nums[right]
        if abs(s - target) < abs(best - target):
            best = s
        if s < target:
            left += 1
        elif s > target:
            right -= 1
        else:
            return s
    return best
'''

        return CodeTemplate(
            name="two_pointer_opposite",
            principle="amortized_invariant",
            description="Two pointers from opposite ends",
            applicability=applicable,
            generate=generate,
        )

    def _sliding_window(self) -> CodeTemplate:
        def applicable(info: dict) -> bool:
            return (
                info.get("optimization_goal") in ("maximize", "minimize", "find")
                and info.get("metric") in ("length", "size")
                and info.get("contiguity") == "contiguous"
            )

        def generate(info: dict) -> str:
            return '''def solve(s, k):
    from collections import defaultdict
    count = defaultdict(int)
    left = 0
    result = 0
    for right in range(len(s)):
        count[s[right]] += 1
        while len(count) > k:
            count[s[left]] -= 1
            if count[s[left]] == 0:
                del count[s[left]]
            left += 1
        result = max(result, right - left + 1)
    return result
'''

        return CodeTemplate(
            name="sliding_window",
            principle="amortized_invariant",
            description="Sliding window with invariant maintenance",
            applicability=applicable,
            generate=generate,
        )

    def _hash_complement(self) -> CodeTemplate:
        def applicable(info: dict) -> bool:
            return (
                info.get("optimization_goal") == "find"
                and info.get("has_target") is True
                and info.get("has_pairs") is True
            )

        def generate(info: dict) -> str:
            return '''def solve(nums, target):
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []
'''

        return CodeTemplate(
            name="hash_complement",
            principle="hashing_fingerprint",
            description="Hash map for complement/pair finding",
            applicability=applicable,
            generate=generate,
        )

    def _hash_frequency(self) -> CodeTemplate:
        def applicable(info: dict) -> bool:
            return (
                info.get("optimization_goal") in ("count", "find", "check")
                and info.get("needs_frequency") is True
            )

        def generate(info: dict) -> str:
            return '''def solve(nums):
    from collections import Counter
    freq = Counter(nums)
    result = []
    for num, count in freq.items():
        if count > len(nums) // 3:
            result.append(num)
    return result
'''

        return CodeTemplate(
            name="hash_frequency",
            principle="hashing_fingerprint",
            description="Hash map frequency counter",
            applicability=applicable,
            generate=generate,
        )

    def _monotonic_stack(self) -> CodeTemplate:
        def applicable(info: dict) -> bool:
            return (
                info.get("optimization_goal") in ("maximize", "minimize", "find")
                and info.get("needs_next_greater") is True
            )

        def generate(info: dict) -> str:
            return '''def solve(nums):
    n = len(nums)
    result = [-1] * n
    stack = []
    for i in range(n):
        while stack and nums[stack[-1]] < nums[i]:
            result[stack.pop()] = nums[i]
        stack.append(i)
    return result
'''

        return CodeTemplate(
            name="monotonic_stack",
            principle="amortized_invariant",
            description="Monotonic stack for next greater element",
            applicability=applicable,
            generate=generate,
        )

    def _graph_bfs(self) -> CodeTemplate:
        def applicable(info: dict) -> bool:
            return (
                info.get("domain") == "graph"
                and info.get("optimization_goal") in ("find", "minimize")
                and info.get("needs_shortest") is not False
            )

        def generate(info: dict) -> str:
            return '''def solve(graph, start, end):
    from collections import deque
    visited = {start}
    queue = deque([(start, 0)])
    while queue:
        node, dist = queue.popleft()
        if node == end:
            return dist
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, dist + 1))
    return -1
'''

        return CodeTemplate(
            name="graph_bfs",
            principle="graph_connectivity",
            description="BFS shortest path in unweighted graph",
            applicability=applicable,
            generate=generate,
        )

    def _graph_dfs(self) -> CodeTemplate:
        def applicable(info: dict) -> bool:
            return (
                info.get("domain") == "graph"
                and info.get("optimization_goal") in ("count", "check", "find")
                and info.get("needs_shortest") is not True
            )

        def generate(info: dict) -> str:
            return '''def solve(graph, start):
    visited = set()
    result = []
    def dfs(node):
        if node in visited:
            return
        visited.add(node)
        result.append(node)
        for neighbor in graph.get(node, []):
            dfs(neighbor)
    dfs(start)
    return result
'''

        return CodeTemplate(
            name="graph_dfs",
            principle="graph_connectivity",
            description="DFS traversal of graph",
            applicability=applicable,
            generate=generate,
        )

    def _prefix_sum(self) -> CodeTemplate:
        def applicable(info: dict) -> bool:
            return (
                info.get("optimization_goal") in ("find", "count", "minimize", "maximize")
                and info.get("needs_range_query") is True
            )

        def generate(info: dict) -> str:
            return '''def solve(nums, queries):
    prefix = [0] * (len(nums) + 1)
    for i in range(len(nums)):
        prefix[i + 1] = prefix[i] + nums[i]
    results = []
    for l, r in queries:
        results.append(prefix[r + 1] - prefix[l])
    return results
'''

        return CodeTemplate(
            name="prefix_sum",
            principle="amortized_invariant",
            description="Prefix sum for range queries",
            applicability=applicable,
            generate=generate,
        )

    def _brute_force(self) -> CodeTemplate:
        def applicable(info: dict) -> bool:
            return True

        def generate(info: dict) -> str:
            return '''def solve(nums):
    best = None
    n = len(nums)
    for i in range(n):
        for j in range(i, n):
            sub = nums[i:j+1]
            val = sum(sub)
            if best is None or val > best:
                best = val
    return best if best is not None else 0
'''

        return CodeTemplate(
            name="brute_force",
            principle="",
            description="Brute force: try all subarrays",
            applicability=applicable,
            generate=generate,
        )
