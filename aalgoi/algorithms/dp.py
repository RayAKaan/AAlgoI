from __future__ import annotations

from typing import Any

from aalgoi.algorithms.base import Algorithm
from aalgoi.algorithms.registry import algorithm
from aalgoi.types import (
    AlgorithmSpec, Complexity, Domain, ProblemSpec, ProblemTask,
)


@algorithm(AlgorithmSpec(
    name="kadane",
    task=ProblemTask.KADANE,
    domain=Domain.DP,
    complexity=Complexity("O(n)", "O(1)", "n", "1"),
    principles=frozenset({"optimal_substructure", "dynamic_programming"}),
    deterministic=True, exact=True,
))
class Kadane(Algorithm):
    def run(self, spec: ProblemSpec) -> Any:
        data = _get_list(spec)
        if not data:
            return 0
        best = curr = data[0]
        for x in data[1:]:
            curr = max(x, curr + x)
            best = max(best, curr)
        return best


@algorithm(AlgorithmSpec(
    name="knapsack_01",
    task=ProblemTask.KNAPSACK_01,
    domain=Domain.DP,
    complexity=Complexity("O(n*W)", "O(W)", "n*W", "W"),
    principles=frozenset({"optimal_substructure", "dynamic_programming"}),
    deterministic=True, exact=True,
))
class Knapsack01(Algorithm):
    def run(self, spec: ProblemSpec) -> Any:
        items, capacity = _get_items_and_capacity(spec)
        n = len(items)
        dp = [0] * (capacity + 1)
        for item in items:
            w = item.get("weight", item.get("wt", 1))
            v = item.get("value", item.get("val", 0))
            for cap in range(capacity, w - 1, -1):
                dp[cap] = max(dp[cap], dp[cap - w] + v)
        selected = []
        cap = capacity
        for item in reversed(items):
            w = item.get("weight", item.get("wt", 1))
            v = item.get("value", item.get("val", 0))
            if cap >= w and dp[cap] == dp[cap - w] + v:
                selected.append(item)
                cap -= w
        return {
            "max_value": dp[capacity],
            "selected": list(reversed(selected)),
            "capacity_used": capacity - cap,
        }


@algorithm(AlgorithmSpec(
    name="coin_change",
    task=ProblemTask.COIN_CHANGE,
    domain=Domain.DP,
    complexity=Complexity("O(n*amount)", "O(amount)", "n*amount", "amount"),
    principles=frozenset({"optimal_substructure", "dynamic_programming"}),
    deterministic=True, exact=True,
))
class CoinChange(Algorithm):
    def run(self, spec: ProblemSpec) -> Any:
        coins, amount = _get_coins_and_amount(spec)
        INF = amount + 1
        dp = [INF] * (amount + 1)
        dp[0] = 0
        for coin in coins:
            for x in range(coin, amount + 1):
                dp[x] = min(dp[x], dp[x - coin] + 1)
        return dp[amount] if dp[amount] != INF else -1


@algorithm(AlgorithmSpec(
    name="lis",
    task=ProblemTask.LIS,
    domain=Domain.DP,
    complexity=Complexity("O(n log n)", "O(n)", "n log n", "n"),
    principles=frozenset({"optimal_substructure", "dynamic_programming", "binary_search"}),
    deterministic=True, exact=True,
))
class LIS(Algorithm):
    def run(self, spec: ProblemSpec) -> Any:
        import bisect
        data = _get_list(spec)
        tails: list[int] = []
        for x in data:
            i = bisect.bisect_left(tails, x)
            if i == len(tails):
                tails.append(x)
            else:
                tails[i] = x
        return len(tails)


def _get_list(spec: ProblemSpec) -> list:
    for val in spec.inputs.values():
        if isinstance(val, list):
            return val
    return []


def _get_items_and_capacity(spec: ProblemSpec) -> tuple[list, int]:
    items = []
    capacity = 0
    for key, val in spec.inputs.items():
        if isinstance(val, list):
            items = val
        elif isinstance(val, (int, float)):
            capacity = int(val)
    return items, capacity


def _get_coins_and_amount(spec: ProblemSpec) -> tuple[list[int], int]:
    coins = []
    amount = 0
    for key, val in spec.inputs.items():
        if isinstance(val, list):
            coins = [int(x) for x in val]
        elif isinstance(val, (int, float)):
            amount = int(val)
    return coins, amount
