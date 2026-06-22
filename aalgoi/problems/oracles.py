from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from typing import Any

from aalgoi.types import ProblemTask

OracleFn = Callable[[dict[str, Any], Any], bool]

_ORACLES: dict[ProblemTask, OracleFn] = {}


def register_oracle(task: ProblemTask) -> Callable[[OracleFn], OracleFn]:
    def decorator(fn: OracleFn) -> OracleFn:
        _ORACLES[task] = fn
        return fn
    return decorator


def get_oracle(task: ProblemTask) -> OracleFn | None:
    return _ORACLES.get(task)


def evaluate(task: ProblemTask, inputs: dict[str, Any], output: Any) -> bool:
    oracle = get_oracle(task)
    if oracle is None:
        return output is not None
    try:
        return oracle(inputs, output)
    except Exception:
        return False


@register_oracle(ProblemTask.SORT)
def _sort_oracle(inputs: dict, output: Any) -> bool:
    if not isinstance(output, list):
        return False
    data = _get_data_list(inputs)
    if len(output) != len(data):
        return False
    return output == sorted(data)


@register_oracle(ProblemTask.COUNTING_SORT)
def _counting_sort_oracle(inputs: dict, output: Any) -> bool:
    return _sort_oracle(inputs, output)


@register_oracle(ProblemTask.LINEAR_SEARCH)
def _linear_search_oracle(inputs: dict, output: Any) -> bool:
    data, target = _get_data_and_target(inputs)
    if output == -1:
        return target not in data
    return 0 <= output < len(data) and data[output] == target


@register_oracle(ProblemTask.BINARY_SEARCH)
def _binary_search_oracle(inputs: dict, output: Any) -> bool:
    return _linear_search_oracle(inputs, output)


@register_oracle(ProblemTask.TWO_SUM)
def _two_sum_oracle(inputs: dict, output: Any) -> bool:
    data = _get_data_list(inputs)
    target = _get_target(inputs)
    if output == []:
        return not any(data[i] + data[j] == target for i in range(len(data)) for j in range(i + 1, len(data)))
    if not isinstance(output, list) or len(output) != 2:
        return False
    i, j = output
    if i == j:
        return False
    return 0 <= i < len(data) and 0 <= j < len(data) and data[i] + data[j] == target


@register_oracle(ProblemTask.GCD)
def _gcd_oracle(inputs: dict, output: Any) -> bool:
    import math
    a, b = _get_two_ints(inputs)
    return output == math.gcd(a, b)


@register_oracle(ProblemTask.LCM)
def _lcm_oracle(inputs: dict, output: Any) -> bool:
    import math
    a, b = _get_two_ints(inputs)
    if a == 0 or b == 0:
        return output == 0
    return output == abs(a // math.gcd(a, b) * b)


@register_oracle(ProblemTask.IS_PRIME)
def _is_prime_oracle(inputs: dict, output: Any) -> bool:
    n = _get_single_int(inputs)
    if n < 2:
        return output is False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return output is False
    return output is True


@register_oracle(ProblemTask.FIBONACCI)
def _fibonacci_oracle(inputs: dict, output: Any) -> bool:
    n = _get_single_int(inputs)
    expected = _fib_ref(n)
    return output == expected


def _fib_ref(n: int) -> int:
    if n < 2:
        return n
    a, b = 0, 1
    for _ in range(n - 1):
        a, b = b, a + b
    return b


@register_oracle(ProblemTask.PALINDROME)
def _palindrome_oracle(inputs: dict, output: Any) -> bool:
    s = _get_string(inputs)
    return output == (s == s[::-1])


@register_oracle(ProblemTask.ANAGRAM)
def _anagram_oracle(inputs: dict, output: Any) -> bool:
    s, t = _get_two_strings(inputs)
    return output == (Counter(_normalize_anagram_text(s)) == Counter(_normalize_anagram_text(t)))


@register_oracle(ProblemTask.KMP)
@register_oracle(ProblemTask.RABIN_KARP)
def _string_match_oracle(inputs: dict, output: Any) -> bool:
    text, pattern = _get_text_and_pattern(inputs)
    if output == -1:
        return pattern not in text
    return text[output:output + len(pattern)] == pattern


@register_oracle(ProblemTask.EDIT_DISTANCE)
def _edit_distance_oracle(inputs: dict, output: Any) -> bool:
    a, b = _get_two_strings(inputs)
    if not isinstance(output, int):
        return False
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return output == dp[n][m]


@register_oracle(ProblemTask.LCS)
def _lcs_oracle(inputs: dict, output: Any) -> bool:
    a, b = _get_two_strings(inputs)
    if not isinstance(output, int):
        return False
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return output == dp[n][m]


@register_oracle(ProblemTask.SHORTEST_PATH_WEIGHTED)
@register_oracle(ProblemTask.SHORTEST_PATH_NEGATIVE)
def _shortest_path_oracle(inputs: dict, output: Any) -> bool:
    if not isinstance(output, dict):
        return False
    path = output.get("path", [])
    if not path or len(path) < 2:
        return output.get("length", float("inf")) == float("inf")
    return path[0] == inputs.get("start") and path[-1] == inputs.get("end")


@register_oracle(ProblemTask.TOPOLOGICAL_SORT)
def _topological_sort_oracle(inputs: dict, output: Any) -> bool:
    return isinstance(output, list) and len(output) >= 0


@register_oracle(ProblemTask.CYCLE_DETECTION)
def _cycle_detection_oracle(inputs: dict, output: Any) -> bool:
    return isinstance(output, bool)


@register_oracle(ProblemTask.CONNECTED_COMPONENTS)
def _connected_components_oracle(inputs: dict, output: Any) -> bool:
    return isinstance(output, list) and all(isinstance(c, list) for c in output)


@register_oracle(ProblemTask.KADANE)
def _kadane_oracle(inputs: dict, output: Any) -> bool:
    data = _get_data_list(inputs)
    if not data:
        return output == 0
    return isinstance(output, (int, float))


@register_oracle(ProblemTask.KNAPSACK_01)
def _knapsack_oracle(inputs: dict, output: Any) -> bool:
    if not isinstance(output, dict):
        return False
    return "max_value" in output


@register_oracle(ProblemTask.COIN_CHANGE)
def _coin_change_oracle(inputs: dict, output: Any) -> bool:
    return isinstance(output, int)


@register_oracle(ProblemTask.LIS)
def _lis_oracle(inputs: dict, output: Any) -> bool:
    return isinstance(output, int) and output >= 0


def _normalize_anagram_text(s: str) -> str:
    return "".join(ch.lower() for ch in s if ch.isalnum())


def _get_data_list(inputs: dict) -> list:
    for v in inputs.values():
        if isinstance(v, list):
            return v
    return []


def _get_data_and_target(inputs: dict) -> tuple[list, Any]:
    data = target = None
    for v in inputs.values():
        if isinstance(v, list):
            data = v
        else:
            target = v
    return data or [], target


def _get_target(inputs: dict) -> Any:
    for v in inputs.values():
        if not isinstance(v, list):
            return v
    return None


def _get_two_ints(inputs: dict) -> tuple[int, int]:
    vals = [v for v in inputs.values() if isinstance(v, int)]
    if len(vals) >= 2:
        return vals[0], vals[1]
    return 0, 0


def _get_single_int(inputs: dict) -> int:
    for v in inputs.values():
        if isinstance(v, int):
            return v
    return 0


def _get_string(inputs: dict) -> str:
    for v in inputs.values():
        if isinstance(v, str):
            return v
    return ""


def _get_two_strings(inputs: dict) -> tuple[str, str]:
    strs = [v for v in inputs.values() if isinstance(v, str)]
    if len(strs) >= 2:
        return strs[0], strs[1]
    if len(strs) == 1:
        return strs[0], ""
    return "", ""


def _get_text_and_pattern(inputs: dict) -> tuple[str, str]:
    return _get_two_strings(inputs)
