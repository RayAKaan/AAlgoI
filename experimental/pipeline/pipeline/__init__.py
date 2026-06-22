"""
aalgoi.pipeline — Pipeline solver and orchestration.

Submodules:
    result:       Result class for all solve operations
    solver:       UniversalSolver
    orchestrator: AAlgoI orchestrator
"""

from aalgoi.pipeline.orchestrator import AAlgoI
from aalgoi.pipeline.result import Result
from aalgoi.pipeline.solver import UniversalSolver

__all__ = ["Result", "UniversalSolver", "AAlgoI"]
