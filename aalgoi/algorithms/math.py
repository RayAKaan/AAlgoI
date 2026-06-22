from __future__ import annotations

import math
from typing import Any

from aalgoi.algorithms.base import Algorithm
from aalgoi.algorithms.registry import algorithm
from aalgoi.types import (
    AlgorithmSpec, Complexity, Domain, ProblemSpec, ProblemTask,
)


@algorithm(AlgorithmSpec(
    name="gcd",
    task=ProblemTask.GCD,
    domain=Domain.MATH,
    complexity=Complexity("O(log min(a,b))", "O(1)", "log n", "1"),
    principles=frozenset({"optimal_substructure"}),
    deterministic=True, exact=True,
))
class GCD(Algorithm):
    def run(self, spec: ProblemSpec) -> Any:
        a, b = _get_two_ints(spec)
        return math.gcd(a, b)


@algorithm(AlgorithmSpec(
    name="lcm",
    task=ProblemTask.LCM,
    domain=Domain.MATH,
    complexity=Complexity("O(log min(a,b))", "O(1)", "log n", "1"),
    principles=frozenset({"optimal_substructure"}),
    deterministic=True, exact=True,
))
class LCM(Algorithm):
    def run(self, spec: ProblemSpec) -> Any:
        a, b = _get_two_ints(spec)
        if a == 0 or b == 0:
            return 0
        return abs(a // math.gcd(a, b) * b)


@algorithm(AlgorithmSpec(
    name="is_prime",
    task=ProblemTask.IS_PRIME,
    domain=Domain.MATH,
    complexity=Complexity("O(√n)", "O(1)", "sqrt n", "1"),
    principles=frozenset({"exhaustive"}),
    deterministic=True, exact=True,
))
class IsPrime(Algorithm):
    def run(self, spec: ProblemSpec) -> Any:
        n = _get_single_int(spec)
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(math.isqrt(n)) + 1, 2):
            if n % i == 0:
                return False
        return True


@algorithm(AlgorithmSpec(
    name="sieve",
    task=ProblemTask.SIEVE,
    domain=Domain.MATH,
    complexity=Complexity("O(n log log n)", "O(n)", "n log log n", "n"),
    principles=frozenset({"exhaustive", "hash_table"}),
    deterministic=True, exact=True,
))
class Sieve(Algorithm):
    def run(self, spec: ProblemSpec) -> Any:
        n = _get_single_int(spec)
        if n < 2:
            return []
        sieve_arr = [True] * (n + 1)
        sieve_arr[0] = sieve_arr[1] = False
        for i in range(2, int(math.isqrt(n)) + 1):
            if sieve_arr[i]:
                step = i
                start = i * i
                sieve_arr[start:n + 1:step] = [False] * ((n - start) // step + 1)
        return [i for i, is_prime in enumerate(sieve_arr) if is_prime]


@algorithm(AlgorithmSpec(
    name="fast_exponentiation",
    task=ProblemTask.FAST_EXPONENTIATION,
    domain=Domain.MATH,
    complexity=Complexity("O(log n)", "O(1)", "log n", "1"),
    principles=frozenset({"divide_conquer"}),
    deterministic=True, exact=True,
))
class FastExponentiation(Algorithm):
    def run(self, spec: ProblemSpec) -> Any:
        base_val, exp = _get_base_and_exp(spec)
        if exp < 0:
            if base_val == 0:
                raise ZeroDivisionError("0 cannot be raised to a negative exponent")
            return 1 / self._pow_positive(base_val, -exp)
        return self._pow_positive(base_val, exp)

    @staticmethod
    def _pow_positive(base_val: int | float, exp: int) -> Any:
        result = 1
        b = base_val
        e = exp
        while e > 0:
            if e & 1:
                result *= b
            b *= b
            e >>= 1
        return result


@algorithm(AlgorithmSpec(
    name="fibonacci",
    task=ProblemTask.FIBONACCI,
    domain=Domain.MATH,
    complexity=Complexity("O(log n)", "O(1)", "log n", "1"),
    principles=frozenset({"optimal_substructure", "divide_conquer"}),
    deterministic=True, exact=True,
))
class Fibonacci(Algorithm):
    def run(self, spec: ProblemSpec) -> Any:
        n = _get_single_int(spec)
        if n < 0:
            raise ValueError("Fibonacci is not defined for negative numbers")
        return _fib_fast_doubling(n)


def _fib_fast_doubling(n: int) -> int:
    if n == 0:
        return 0
    def fib_pair(k: int) -> tuple[int, int]:
        if k == 0:
            return (0, 1)
        a, b = fib_pair(k >> 1)
        c = a * ((b << 1) - a)
        d = a * a + b * b
        if k & 1:
            return (d, c + d)
        return (c, d)
    return fib_pair(n)[0]


def _get_two_ints(spec: ProblemSpec) -> tuple[int, int]:
    vals = [v for v in spec.inputs.values() if isinstance(v, int)]
    if len(vals) >= 2:
        return vals[0], vals[1]
    return (0, 0)


def _get_single_int(spec: ProblemSpec) -> int:
    for v in spec.inputs.values():
        if isinstance(v, int):
            return v
    return 0


def _get_base_and_exp(spec: ProblemSpec) -> tuple[int, int]:
    vals = [v for v in spec.inputs.values() if isinstance(v, int)]
    if len(vals) >= 2:
        return vals[0], vals[1]
    return (0, 0)
