from __future__ import annotations

from typing import Any

import numpy as np

from aalgoi.types import ProblemSpec, ProblemTask


class FeatureExtractor:
    def extract(self, spec: ProblemSpec) -> np.ndarray:
        vec = np.zeros(32, dtype=np.float32)
        all_tasks = sorted(ProblemTask, key=lambda t: t.value)
        task_idx = all_tasks.index(spec.task) if spec.task in all_tasks else 0
        if task_idx < 16:
            vec[task_idx] = 1.0
        vec[16] = len(spec.inputs) / 10.0
        data_size = 0
        for v in spec.inputs.values():
            if isinstance(v, (list, str)):
                data_size = len(v)
                break
        vec[17] = np.log10(max(data_size, 1)) / 6.0
        vec[18] = float(spec.constraints.allow_approximate)
        vec[19] = 1.0 if spec.examples else 0.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec
