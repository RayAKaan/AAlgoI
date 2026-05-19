
import unittest
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.problem_spec import ProblemSpec, ProblemType, Objective, Constraint
from core.problem_library import ProblemLibrary
from algorithms.primitives import (
    PRIMITIVES, IteratePrimitive, MapPrimitive, FilterPrimitive,
    ReducePrimitive, ScanPrimitive, PartitionPrimitive,
    BinarySearchPrimitive, LinearSearchPrimitive,
    GreedyPrimitive, DynamicProgrammingPrimitive,
    GradientDescentPrimitive, compose_pipeline, get_composable_chain
)


class TestProblemSpecCreation(unittest.TestCase):
    def test_sorting_spec(self):
        spec = ProblemSpec(
            name="sort_numbers",
            problem_type=ProblemType.OPTIMIZATION,
            inputs={"numbers": {"type": "list[int]"}},
            outputs={"sorted": {"type": "list[int]"}},
            objectives=[Objective("minimize", "comparisons")]
        )
        self.assertEqual(spec.name, "sort_numbers")
        self.assertEqual(spec.problem_type, ProblemType.OPTIMIZATION)
        self.assertIn("numbers", spec.inputs)
        self.assertIn("sorted", spec.outputs)

    def test_tsp_spec(self):
        tsp = ProblemSpec(
            name="traveling_salesman",
            problem_type=ProblemType.OPTIMIZATION,
            inputs={"cities": {"type": "list[tuple]"}},
            outputs={"tour": {"type": "list[int]"}},
            constraints=["visit_each_once", "return_to_start"],
            objectives=[Objective("minimize", "total_distance")]
        )
        self.assertEqual(len(tsp.constraints), 2)
        self.assertTrue(any("visit" in str(c) for c in tsp.constraints))

    def test_to_vector_shape(self):
        spec = ProblemSpec(
            name="test",
            inputs={"x": {"type": "int"}},
            outputs={"y": {"type": "int"}}
        )
        vec = spec.to_vector()
        self.assertEqual(vec.shape, (128,))
        self.assertAlmostEqual(np.linalg.norm(vec), 1.0, places=5)

    def test_infer_problem_type_routing(self):
        spec = ProblemSpec(
            name="find_shortest_route",
            inputs={"cities": {"type": "list"}},
            outputs={"path": {"type": "list"}},
            constraints=["visit all cities", "shortest distance"]
        )
        inferred = spec.infer_problem_type()
        self.assertEqual(inferred, ProblemType.ROUTING)

    def test_infer_problem_type_sorting(self):
        spec = ProblemSpec(
            name="sort these numbers",
            inputs={"data": {"type": "list"}},
            outputs={"sorted": {"type": "list"}}
        )
        inferred = spec.infer_problem_type()
        self.assertEqual(inferred, ProblemType.SORTING)

    def test_validate_success(self):
        spec = ProblemSpec(
            name="test",
            inputs={"x": {"type": "int"}},
            outputs={"y": {"type": "int"}},
            objectives=[Objective("minimize", "time")]
        )
        valid, errors = spec.validate()
        self.assertTrue(valid)
        self.assertEqual(len(errors), 0)

    def test_validate_missing_name(self):
        spec = ProblemSpec(
            name="",
            inputs={"x": {"type": "int"}},
            outputs={"y": {"type": "int"}}
        )
        valid, errors = spec.validate()
        self.assertFalse(valid)
        self.assertTrue(any("name" in e for e in errors))

    def test_to_dict_roundtrip(self):
        spec = ProblemSpec(
            name="test",
            problem_type=ProblemType.SEARCH,
            inputs={"query": {"type": "str"}},
            outputs={"result": {"type": "list"}},
            constraints=["fast"],
            objectives=[Objective("minimize", "time")]
        )
        d = spec.to_dict()
        restored = ProblemSpec.from_dict(d)
        self.assertEqual(restored.name, spec.name)
        self.assertEqual(restored.problem_type, spec.problem_type)

    def test_get_signature(self):
        spec = ProblemSpec(
            name="test",
            inputs={"a": {"type": "int"}},
            outputs={"b": {"type": "int"}},
            objectives=[Objective("minimize", "x")]
        )
        sig = spec.get_signature()
        self.assertIn("optimization", sig)
        self.assertIn("a:int", sig)
        self.assertIn("b:int", sig)


class TestProblemLibrary(unittest.TestCase):
    def setUp(self):
        self.library = ProblemLibrary()

    def test_store_and_find_similar(self):
        sorting = ProblemSpec(
            name="sort_numbers",
            inputs={"numbers": {"type": "list[int]"}},
            outputs={"sorted": {"type": "list[int]"}},
            objectives=[Objective("minimize", "comparisons")]
        )
        self.library.store_solution(sorting, ["timsort"], {"score": 0.95})

        similar = self.library.find_similar(sorting, top_k=1)
        self.assertEqual(len(self.library.problems), 1)

    def test_store_same_problem_updates(self):
        spec_a = ProblemSpec(
            name="test_a",
            inputs={"x": {"type": "list[int]"}},
            outputs={"sorted": {"type": "list[int]"}}
        )
        spec_b = ProblemSpec(
            name="test_b",
            inputs={"x": {"type": "list[str]"}},
            outputs={"filtered": {"type": "list[str]"}}
        )
        self.library.store_solution(spec_a, ["algo_a"], {"score": 0.8})
        self.library.store_solution(spec_b, ["algo_b"], {"score": 0.9})

        self.assertEqual(len(self.library.problems), 2)

    def test_get_best_algorithms_empty(self):
        spec = ProblemSpec(
            name="novel",
            inputs={"x": {"type": "int"}},
            outputs={"y": {"type": "int"}}
        )
        best = self.library.get_best_algorithms(spec)
        self.assertEqual(len(best), 0)

    def test_get_stats_empty(self):
        stats = self.library.get_stats()
        self.assertEqual(stats["total_problems"], 0)


class TestPrimitives(unittest.TestCase):
    def test_iterate_primitive(self):
        prim = IteratePrimitive()
        result = prim.process([1, 2, 3])
        self.assertEqual(result, [1, 2, 3])

    def test_map_primitive(self):
        prim = MapPrimitive(transform_fn=lambda x: x * 2)
        result = prim.process([1, 2, 3])
        self.assertEqual(result, [2, 4, 6])

    def test_filter_primitive(self):
        prim = FilterPrimitive(predicate_fn=lambda x: x > 2)
        result = prim.process([1, 2, 3, 4])
        self.assertEqual(result, [3, 4])

    def test_reduce_primitive(self):
        prim = ReducePrimitive(reduce_fn=lambda a, b: a + b, initial=0)
        result = prim.process([1, 2, 3, 4])
        self.assertEqual(result, 10)

    def test_scan_primitive(self):
        prim = ScanPrimitive()
        result = prim.process([1, 2, 3])
        self.assertEqual(result, [1, 3, 6])

    def test_partition_primitive(self):
        prim = PartitionPrimitive()
        result = prim.process([3, 1, 4, 1, 5])
        self.assertEqual(len(result), 3)

    def test_binary_search_found(self):
        prim = BinarySearchPrimitive(target=5)
        result = prim.process([1, 3, 5, 7, 9])
        self.assertEqual(result, 2)

    def test_binary_search_not_found(self):
        prim = BinarySearchPrimitive(target=6)
        result = prim.process([1, 3, 5, 7, 9])
        self.assertEqual(result, -1)

    def test_linear_search(self):
        prim = LinearSearchPrimitive(target=3)
        result = prim.process([1, 2, 3, 4])
        self.assertEqual(result, 2)

    def test_greedy_primitive(self):
        prim = GreedyPrimitive()
        result = prim.process([3, 1, 2])
        self.assertEqual(result, [1, 2, 3])

    def test_dynamic_programming_primitive(self):
        prim = DynamicProgrammingPrimitive(dp_fn=lambda data: max(data))
        result = prim.process([1, 5, 3])
        self.assertEqual(result, 5)

    def test_dp_default_fallback(self):
        prim = DynamicProgrammingPrimitive()
        result = prim.process([1, 5, 3])
        self.assertEqual(result, 5)

    def test_gradient_descent_primitive(self):
        prim = GradientDescentPrimitive()
        result = prim.process([1.0, 2.0, 3.0])
        self.assertIsNotNone(result)

    def test_validate_output(self):
        prim = BinarySearchPrimitive(target=3)
        result = prim.process([1, 2, 3, 4])
        self.assertTrue(prim.validate_output([1, 2, 3, 4], result))


class TestPrimitiveComposability(unittest.TestCase):
    def test_map_filter_chain(self):
        map_p = MapPrimitive()
        filter_p = FilterPrimitive()
        self.assertTrue(map_p.can_compose_with(filter_p))

    def test_iterate_map_chain(self):
        it = IteratePrimitive()
        mp = MapPrimitive()
        self.assertTrue(it.can_compose_with(mp))

    def test_incompatible_primitives(self):
        search = BinarySearchPrimitive()
        self.assertFalse(search.can_compose_with(search))

    def test_compose_pipeline(self):
        pipeline = compose_pipeline(["iterate", "map", "filter"])
        self.assertIsNotNone(pipeline)
        self.assertEqual(len(pipeline), 3)

    def test_compose_invalid_pipeline(self):
        pipeline = compose_pipeline(["binary_search", "iterate"])
        self.assertIsNone(pipeline)

    def test_get_composable_chain(self):
        chain = get_composable_chain("iterate", "reduce")
        self.assertIsNotNone(chain)
        self.assertGreaterEqual(len(chain), 2)

    def test_primitive_registry(self):
        self.assertIn("iterate", PRIMITIVES)
        self.assertIn("map", PRIMITIVES)
        self.assertIn("filter", PRIMITIVES)
        self.assertIn("reduce", PRIMITIVES)
        self.assertIn("binary_search", PRIMITIVES)
        self.assertIn("greedy", PRIMITIVES)
        self.assertEqual(len(PRIMITIVES), 25)

    def test_primitive_describe(self):
        prim = IteratePrimitive()
        info = prim.describe()
        self.assertIn("time_complexity", info)
        self.assertIn("space_complexity", info)
        self.assertIn("best_for", info)
        self.assertIn("combines_well_with", info)


if __name__ == "__main__":
    unittest.main()
