from __future__ import annotations

from aalgoi.eval.benchmark_suite import core_v1
from aalgoi.types import BenchmarkReport


def run_benchmark(suite: str = "core-v1") -> BenchmarkReport:
    bs = core_v1() if suite == "core-v1" else None
    if bs is None:
        raise ValueError(f"Unknown benchmark suite: {suite}")
    return bs.run()
