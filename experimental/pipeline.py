"""
aalgoi.pipeline — Pipeline solver and orchestration.

This module re-exports from the aalgoi.pipeline subpackage for
backward compatibility.

Usage:
    from aalgoi.pipeline import Result, UniversalSolver, AAlgoI
"""

from aalgoi.pipeline.orchestrator import AAlgoI
from aalgoi.pipeline.result import Result
from aalgoi.pipeline.solver import UniversalSolver

__all__ = ["Result", "UniversalSolver", "AAlgoI"]
