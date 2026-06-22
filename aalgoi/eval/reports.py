from __future__ import annotations

from dataclasses import dataclass

from aalgoi.types import BenchmarkReport as BenchmarkReportType


@dataclass
class BenchmarkReport:
    data: BenchmarkReportType

    def __str__(self) -> str:
        return self._format()

    def _format(self) -> str:
        lines = [f"Suite: {self.data.suite}"]
        lines.append(f"Total: {self.data.total}")
        lines.append(f"Correct: {self.data.correct}")
        lines.append(f"Failed: {self.data.failed}")
        lines.append(f"Errors: {self.data.errors}")
        lines.append(f"Accuracy: {self.data.accuracy:.1%}")
        lines.append(f"Time: {self.data.time_ms:.0f}ms")
        if self.data.by_task:
            lines.append("")
            lines.append("By task:")
            for task_key, d in sorted(self.data.by_task.items()):
                pct = d["passed"] / max(d["total"], 1) * 100
                lines.append(f"  {task_key}: {d['passed']}/{d['total']} ({pct:.0f}%)")
        return "\n".join(lines)


def format_report(data: BenchmarkReportType) -> str:
    return str(BenchmarkReport(data))
