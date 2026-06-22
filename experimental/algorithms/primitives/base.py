from typing import Any

import numpy as np

from aalgoi.algorithms.base import Algorithm


class Primitive(Algorithm):
    time_complexity: str = "O(1)"
    space_complexity: str = "O(1)"
    best_for: list[str] = []
    combines_well_with: list[str] = []
    input_type: str = "any"
    output_type: str = "any"

    def __init__(self) -> None:
        super().__init__()
        self.transform_fn = lambda x: x.get('data') if isinstance(x, dict) else x

    @staticmethod
    def _unwrap_data(data: Any) -> Any:
        if isinstance(data, dict) and 'data' in data:
            return data['data']
        return data

    def can_compose_with(self, other: "Primitive") -> bool:
        return self.output_type == other.input_type or other.input_type == "any" or self.output_type == "any"

    def describe(self) -> dict[str, Any]:
        info = super().describe()
        info.update({
            "time_complexity": self.time_complexity,
            "space_complexity": self.space_complexity,
            "best_for": self.best_for,
            "combines_well_with": self.combines_well_with,
            "input_type": self.input_type,
            "output_type": self.output_type
        })
        return info
