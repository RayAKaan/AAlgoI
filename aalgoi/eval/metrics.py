from __future__ import annotations

from aalgoi.types import BenchmarkReport


class Metrics:
    @staticmethod
    def accuracy(report: BenchmarkReport) -> float:
        return report.accuracy

    @staticmethod
    def precision(report: BenchmarkReport, task: str | None = None) -> float:
        if task and task in report.by_task:
            d = report.by_task[task]
            total = d["total"]
            passed = d["passed"]
            return passed / max(total, 1)
        return report.accuracy

    @staticmethod
    def summary(report: BenchmarkReport) -> str:
        pct = int(report.accuracy * 100)
        return f"{report.suite}: {report.correct}/{report.total} correct ({pct}%) in {report.time_ms:.0f}ms"
