from __future__ import annotations

from typing import Any

from aalgoi.algorithms.base import Algorithm
from aalgoi.algorithms.registry import algorithm
from aalgoi.types import (
    AlgorithmSpec, Complexity, Domain, ProblemSpec, ProblemTask,
)


@algorithm(AlgorithmSpec(
    name="gaussian_blur",
    task=ProblemTask.LINEAR_SEARCH,
    domain=Domain.IMAGE,
    complexity=Complexity("O(n)", "O(n)", "n", "n"),
    principles=frozenset({"convolution"}),
    deterministic=True, exact=False,
    tags=frozenset({"image", "optional"}),
))
class GaussianBlur(Algorithm):
    def run(self, spec: ProblemSpec) -> Any:
        from aalgoi.errors import OptionalDependencyMissing
        try:
            import numpy as np
            from scipy.ndimage import gaussian_filter
        except ImportError:
            raise OptionalDependencyMissing("image", "scipy")
        img = spec.inputs.get("image", spec.inputs.get("data"))
        if img is None:
            return []
        sigma = spec.inputs.get("sigma", 1.0)
        return gaussian_filter(np.array(img, dtype=float), sigma=sigma).tolist()


@algorithm(AlgorithmSpec(
    name="edge_detection",
    task=ProblemTask.LINEAR_SEARCH,
    domain=Domain.IMAGE,
    complexity=Complexity("O(n)", "O(n)", "n", "n"),
    principles=frozenset({"convolution"}),
    deterministic=True, exact=False,
    tags=frozenset({"image", "optional"}),
))
class EdgeDetection(Algorithm):
    def run(self, spec: ProblemSpec) -> Any:
        from aalgoi.errors import OptionalDependencyMissing
        try:
            import numpy as np
            from scipy.ndimage import sobel
        except ImportError:
            raise OptionalDependencyMissing("image", "scipy")
        img = spec.inputs.get("image", spec.inputs.get("data"))
        if img is None:
            return []
        arr = np.array(img, dtype=float)
        edges = np.hypot(sobel(arr, axis=0), sobel(arr, axis=1))
        return edges.tolist()
