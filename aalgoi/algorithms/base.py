
import copy
import time
from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class Algorithm(ABC):

    def __init__(self):
        self.name = getattr(self, 'name', "base")
        self.time_complexity: str = "O(1)"
        self.space_complexity: str = "O(1)"
        self.tags: list[str] = []
        self.best_for: list[str] = []
        self.patterns: list[str] = []
        self.problem_types: list[str] = []
        self.params: dict[str, Any] = {}

        self.modifications: list[str] = []
        self.last_execution_time: float | None = None
        self.last_memory_usage: float | None = None

    @abstractmethod
    def process(self, data: Any) -> Any:
        pass

    def validate_output(self, input_data: Any, output_data: Any) -> bool:
        if hasattr(input_data, '__len__') and hasattr(output_data, '__len__'):
            if len(input_data) != len(output_data):
                return False

        if type(input_data) is not type(output_data):  # noqa: E721
            if not (isinstance(input_data, (list, tuple, np.ndarray, str)) and
                    isinstance(output_data, (list, tuple, np.ndarray, str))):
                return False

        return True

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {
            'name': self.name,
            'time_complexity': self.time_complexity,
            'space_complexity': self.space_complexity,
            'tags': self.tags,
            'best_for': self.best_for,
        }

    def set_params(self, **params):
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "tags": self.tags,
            "best_for": self.best_for,
            "time_complexity": self.time_complexity,
            "space_complexity": self.space_complexity,
            "patterns": self.patterns,
            "problem_types": self.problem_types,
        }

    def describe(self) -> dict[str, Any]:
        return self.metadata()

    def clone(self):
        return copy.deepcopy(self)

    def _timed_process(self, data: Any):
        start = time.perf_counter()
        result = self.process(data)
        elapsed = time.perf_counter() - start
        self.last_execution_time = elapsed * 1000
        return result, elapsed
