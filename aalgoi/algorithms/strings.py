from __future__ import annotations

from collections import Counter
from typing import Any

from aalgoi.algorithms.base import Algorithm
from aalgoi.algorithms.registry import algorithm
from aalgoi.types import (
    AlgorithmSpec, Complexity, Domain, ProblemSpec, ProblemTask,
)


@algorithm(AlgorithmSpec(
    name="palindrome",
    task=ProblemTask.PALINDROME,
    domain=Domain.STRINGS,
    complexity=Complexity("O(n)", "O(1)", "n", "1"),
    principles=frozenset({"two_pointers"}),
    deterministic=True, exact=True,
))
class Palindrome(Algorithm):
    def run(self, spec: ProblemSpec) -> Any:
        s = _get_string(spec)
        i, j = 0, len(s) - 1
        while i < j:
            if s[i] != s[j]:
                return False
            i += 1
            j -= 1
        return True


@algorithm(AlgorithmSpec(
    name="anagram",
    task=ProblemTask.ANAGRAM,
    domain=Domain.STRINGS,
    complexity=Complexity("O(n)", "O(k)", "n", "k"),
    principles=frozenset({"hash_table"}),
    deterministic=True, exact=True,
))
class Anagram(Algorithm):
    def run(self, spec: ProblemSpec) -> Any:
        s, t = _get_two_strings(spec)
        return Counter(s) == Counter(t)


@algorithm(AlgorithmSpec(
    name="kmp",
    task=ProblemTask.KMP,
    domain=Domain.STRINGS,
    complexity=Complexity("O(n+m)", "O(m)", "n+m", "m"),
    principles=frozenset({"hash_table", "optimal_substructure"}),
    deterministic=True, exact=True,
))
class KMP(Algorithm):
    def run(self, spec: ProblemSpec) -> Any:
        text, pattern = _get_text_and_pattern(spec)
        n, m = len(text), len(pattern)
        if m == 0:
            return 0
        lps = _build_lps(pattern)
        i = j = 0
        while i < n:
            if text[i] == pattern[j]:
                i += 1
                j += 1
                if j == m:
                    return i - j
            else:
                if j != 0:
                    j = lps[j - 1]
                else:
                    i += 1
        return -1


def _build_lps(pattern: str) -> list[int]:
    m = len(pattern)
    lps = [0] * m
    length = 0
    i = 1
    while i < m:
        if pattern[i] == pattern[length]:
            length += 1
            lps[i] = length
            i += 1
        else:
            if length != 0:
                length = lps[length - 1]
            else:
                lps[i] = 0
                i += 1
    return lps


@algorithm(AlgorithmSpec(
    name="rabin_karp",
    task=ProblemTask.RABIN_KARP,
    domain=Domain.STRINGS,
    complexity=Complexity("O(n+m)", "O(1)", "n+m", "1"),
    principles=frozenset({"hash_table"}),
    deterministic=True, exact=True,
))
class RabinKarp(Algorithm):
    def run(self, spec: ProblemSpec) -> Any:
        text, pattern = _get_text_and_pattern(spec)
        n, m = len(text), len(pattern)
        if m == 0:
            return 0
        if m > n:
            return -1
        d = 256
        q = 101
        h = pow(d, m - 1, q)
        p_hash = t_hash = 0
        for i in range(m):
            p_hash = (d * p_hash + ord(pattern[i])) % q
            t_hash = (d * t_hash + ord(text[i])) % q
        for i in range(n - m + 1):
            if p_hash == t_hash:
                if text[i:i + m] == pattern:
                    return i
            if i < n - m:
                t_hash = (d * (t_hash - ord(text[i]) * h) + ord(text[i + m])) % q
                if t_hash < 0:
                    t_hash += q
        return -1


@algorithm(AlgorithmSpec(
    name="edit_distance",
    task=ProblemTask.EDIT_DISTANCE,
    domain=Domain.STRINGS,
    complexity=Complexity("O(n*m)", "O(n*m)", "n*m", "n*m"),
    principles=frozenset({"optimal_substructure", "dynamic_programming"}),
    deterministic=True, exact=True,
))
class EditDistance(Algorithm):
    def run(self, spec: ProblemSpec) -> Any:
        a, b = _get_two_strings_from_inputs(spec)
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
        return dp[n][m]


@algorithm(AlgorithmSpec(
    name="lcs",
    task=ProblemTask.LCS,
    domain=Domain.STRINGS,
    complexity=Complexity("O(n*m)", "O(n*m)", "n*m", "n*m"),
    principles=frozenset({"optimal_substructure", "dynamic_programming"}),
    deterministic=True, exact=True,
))
class LCS(Algorithm):
    def run(self, spec: ProblemSpec) -> Any:
        a, b = _get_two_strings_from_inputs(spec)
        n, m = len(a), len(b)
        dp = [[0] * (m + 1) for _ in range(n + 1)]
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if a[i - 1] == b[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        return dp[n][m]


def _get_string(spec: ProblemSpec) -> str:
    for v in spec.inputs.values():
        if isinstance(v, str):
            return v
    return ""


def _get_two_strings(spec: ProblemSpec) -> tuple[str, str]:
    strs = [v for v in spec.inputs.values() if isinstance(v, str)]
    if len(strs) >= 2:
        return strs[0], strs[1]
    if len(strs) == 1:
        return strs[0], ""
    return "", ""


def _get_two_strings_from_inputs(spec: ProblemSpec) -> tuple[str, str]:
    return _get_two_strings(spec)


def _get_text_and_pattern(spec: ProblemSpec) -> tuple[str, str]:
    return _get_two_strings(spec)
