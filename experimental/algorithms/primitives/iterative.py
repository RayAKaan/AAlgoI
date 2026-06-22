from typing import Any

from aalgoi.algorithms.primitives.base import Primitive


class IteratePrimitive(Primitive):
    name = "iterate"
    tags = ["iteration", "loop", "basic"]
    time_complexity = "O(n)"
    space_complexity = "O(1)"
    input_type = "iterable"
    output_type = "iterable"

    def process(self, data: Any) -> Any:
        if hasattr(data, '__iter__'):
            return list(data)
        return data

    def validate_output(self, input_data: Any, output_data: Any) -> bool:
        if output_data is None:
            return False
        return True


class MapPrimitive(Primitive):
    name = "map"
    tags = ["iteration", "transformation", "functional"]
    time_complexity = "O(n)"
    space_complexity = "O(n)"
    input_type = "iterable"
    output_type = "iterable"
    combines_well_with = ["filter", "reduce"]

    def __init__(self, transform_fn: Any = None) -> None:
        super().__init__()
        self.transform_fn = transform_fn

    def process(self, data: Any) -> Any:
        if isinstance(data, list) and self.transform_fn:
            return [self.transform_fn(x) for x in data]
        if isinstance(data, list):
            return data
        return data


class FilterPrimitive(Primitive):
    name = "filter"
    tags = ["iteration", "selection", "functional"]
    time_complexity = "O(n)"
    space_complexity = "O(n)"
    input_type = "iterable"
    output_type = "iterable"
    combines_well_with = ["map", "reduce"]

    def __init__(self, predicate_fn: Any = None) -> None:
        super().__init__()
        self.predicate_fn = predicate_fn

    def process(self, data: Any) -> Any:
        if isinstance(data, list) and self.predicate_fn:
            return [x for x in data if self.predicate_fn(x)]
        if isinstance(data, list):
            return data
        return data


class ReducePrimitive(Primitive):
    name = "reduce"
    tags = ["iteration", "aggregation", "functional"]
    time_complexity = "O(n)"
    space_complexity = "O(1)"
    input_type = "iterable"
    output_type = "scalar"
    combines_well_with = ["map", "filter"]

    def __init__(self, reduce_fn: Any = None, initial: Any = None) -> None:
        super().__init__()
        self.reduce_fn = reduce_fn
        self.initial = initial

    def process(self, data: Any) -> Any:
        if isinstance(data, list) and self.reduce_fn:
            result = self.initial
            for x in data:
                result = self.reduce_fn(result, x) if result is not None else x
            return result
        if isinstance(data, list):
            return len(data)
        return data


class ScanPrimitive(Primitive):
    name = "scan"
    tags = ["iteration", "accumulation", "prefix"]
    time_complexity = "O(n)"
    space_complexity = "O(n)"
    input_type = "iterable"
    output_type = "iterable"
    combines_well_with = ["map", "filter"]

    def process(self, data: Any) -> Any:
        if isinstance(data, list):
            result = []
            acc = 0
            for x in data:
                if isinstance(x, (int, float)):
                    acc += x
                result.append(acc)
            return result
        return data
