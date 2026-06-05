
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import time
import copy
import numpy as np

class Algorithm(ABC):

    def __init__(self):
        self.name = getattr(self, 'name', "base")
        self.time_complexity: str = "O(1)"
        self.space_complexity: str = "O(1)"
        self.tags: List[str] = []
        self.best_for: List[str] = []
        self.patterns: List[str] = []
        self.problem_types: List[str] = []
        self.params: Dict[str, Any] = {}

        self.modifications: List[str] = []
        self.last_execution_time: Optional[float] = None
        self.last_memory_usage: Optional[float] = None

    @abstractmethod
    def process(self, data: Any) -> Any:
        pass

    def validate_output(self, input_data: Any, output_data: Any) -> bool:
        if hasattr(input_data, '__len__') and hasattr(output_data, '__len__'):
            if len(input_data) != len(output_data):
                return False

        if type(input_data) != type(output_data):
            if not (isinstance(input_data, (list, tuple, np.ndarray)) and
                    isinstance(output_data, (list, tuple, np.ndarray))):
                return False

        return True

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
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

    def metadata(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "tags": self.tags,
            "best_for": self.best_for,
            "time_complexity": self.time_complexity,
            "space_complexity": self.space_complexity,
            "patterns": self.patterns,
            "problem_types": self.problem_types,
        }

    def describe(self) -> Dict[str, Any]:
        return self.metadata()

    def clone(self):
        return copy.deepcopy(self)

    def _timed_process(self, data: Any):
        start = time.perf_counter()
        result = self.process(data)
        elapsed = time.perf_counter() - start
        self.last_execution_time = elapsed * 1000
        return result, elapsed
