from typing import Any

from aalgoi.algorithms.primitives.base import Primitive


class LongestCommonSubsequencePrimitive(Primitive):
    name = "lcs"
    tags = ["string", "dynamic", "subsequence"]
    time_complexity = "O(n*m)"
    space_complexity = "O(n*m)"
    input_type = "iterable"
    output_type = "scalar"
    best_for = ["sequence_matching", "diff", "bioinformatics"]
    combines_well_with = ["dynamic_programming", "string_match"]

    def process(self, data: Any) -> Any:
        if isinstance(data, (tuple, list)) and len(data) == 2:
            a, b = data[0], data[1]
            n, m = len(a), len(b)
            dp = [[0] * (m + 1) for _ in range(n + 1)]
            for i in range(1, n + 1):
                for j in range(1, m + 1):
                    if a[i - 1] == b[j - 1]:
                        dp[i][j] = dp[i - 1][j - 1] + 1
                    else:
                        dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
            return dp[n][m]
        return data


class RabinKarpPrimitive(Primitive):
    name = "rabin_karp"
    tags = ["string", "pattern", "hashing"]
    time_complexity = "O(n + m)"
    space_complexity = "O(1)"
    input_type = "iterable"
    output_type = "scalar"
    best_for = ["pattern_matching", "string_search", "plagiarism"]
    combines_well_with = ["string_match", "lcs"]

    def __init__(self, pattern: Any = None) -> None:
        super().__init__()
        self.pattern = pattern

    def process(self, data: Any) -> Any:
        if isinstance(data, str) and self.pattern:
            n, m = len(data), len(self.pattern)
            if m > n:
                return -1
            d, q = 256, 101
            p_hash = 0
            t_hash = 0
            h = pow(d, m - 1) % q
            for i in range(m):
                p_hash = (d * p_hash + ord(self.pattern[i])) % q
                t_hash = (d * t_hash + ord(data[i])) % q
            for i in range(n - m + 1):
                if p_hash == t_hash:
                    if data[i:i + m] == self.pattern:
                        return i
                if i < n - m:
                    t_hash = (d * (t_hash - ord(data[i]) * h) + ord(data[i + m])) % q
                    if t_hash < 0:
                        t_hash += q
            return -1
        return data

    def validate_output(self, input_data: Any, output_data: Any) -> bool:
        if output_data is None:
            return False
        if isinstance(output_data, int) and self.pattern is not None:
            if output_data == -1:
                return True
            return 0 <= output_data < len(input_data) and input_data[output_data:output_data + len(self.pattern)] == self.pattern
        return True
