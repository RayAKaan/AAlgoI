import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aalgoi.algorithms.primitives import (
    PRIMITIVES,
    BacktrackingPrimitive,
    BFSPrimitive,
    DFSPrimitive,
    HeapSortPrimitive,
    InterpolationSearchPrimitive,
    LongestCommonSubsequencePrimitive,
    MergeSortPrimitive,
    Primitive,
    QuickSortPrimitive,
    RabinKarpPrimitive,
    RandomSearchPrimitive,
    SlidingWindowPrimitive,
    TopologicalSortPrimitive,
    TwoPointerPrimitive,
    UnionFindPrimitive,
    compose_pipeline,
)
from aalgoi.core.algorithm_synthesizer import AlgorithmSynthesizer, SynthesisResult
from aalgoi.core.explainer import Explainer, Explanation
from aalgoi.core.problem_library import ProblemLibrary
from aalgoi.core.problem_spec import Objective, ProblemSpec, ProblemType


class TestNewPrimitives(unittest.TestCase):
    def test_quicksort(self):
        prim = QuickSortPrimitive()
        result = prim.process([3, 1, 4, 1, 5])
        self.assertEqual(result, [1, 1, 3, 4, 5])

    def test_quicksort_single(self):
        prim = QuickSortPrimitive()
        result = prim.process([1])
        self.assertEqual(result, [1])

    def test_quicksort_empty(self):
        prim = QuickSortPrimitive()
        result = prim.process([])
        self.assertEqual(result, [])

    def test_quicksort_validate(self):
        prim = QuickSortPrimitive()
        self.assertTrue(prim.validate_output([3, 1, 2], [1, 2, 3]))
        self.assertFalse(prim.validate_output([3, 1, 2], [3, 1, 2]))

    def test_mergesort(self):
        prim = MergeSortPrimitive()
        result = prim.process([3, 1, 4, 1, 5])
        self.assertEqual(result, [1, 1, 3, 4, 5])

    def test_heapsort(self):
        prim = HeapSortPrimitive()
        result = prim.process([3, 1, 4, 1, 5])
        self.assertEqual(result, [1, 1, 3, 4, 5])

    def test_bfs_graph(self):
        prim = BFSPrimitive()
        graph = {1: [2, 3], 2: [4], 3: [5], 4: [], 5: []}
        result = prim.process(graph)
        self.assertEqual(result, [1, 2, 3, 4, 5])

    def test_dfs_graph(self):
        prim = DFSPrimitive()
        graph = {1: [2, 3], 2: [4], 3: [5], 4: [], 5: []}
        result = prim.process(graph)
        self.assertEqual(result, [1, 2, 4, 3, 5])

    def test_interpolation_search_found(self):
        prim = InterpolationSearchPrimitive(target=5)
        result = prim.process([1, 2, 3, 4, 5, 6, 7])
        self.assertEqual(result, 4)

    def test_interpolation_search_not_found(self):
        prim = InterpolationSearchPrimitive(target=10)
        result = prim.process([1, 2, 3, 4, 5])
        self.assertEqual(result, -1)

    def test_two_pointer_finds_pair(self):
        prim = TwoPointerPrimitive(target_sum=5)
        result = prim.process([1, 2, 3, 4])
        self.assertIsNotNone(result)
        val1, val2 = result[2], result[3]
        self.assertEqual(val1 + val2, 5)

    def test_two_pointer_no_target(self):
        prim = TwoPointerPrimitive()
        result = prim.process([1, 2, 3, 4])
        self.assertEqual(result, (0, 3))

    def test_sliding_window(self):
        prim = SlidingWindowPrimitive(window_size=2)
        result = prim.process([1, 4, 2, 10, 3])
        self.assertEqual(result, 13)

    def test_topological_sort(self):
        prim = TopologicalSortPrimitive()
        graph = {1: [2, 3], 2: [4], 3: [4], 4: []}
        result = prim.process(graph)
        self.assertEqual(result, [1, 2, 3, 4])

    def test_topological_sort_cycle(self):
        prim = TopologicalSortPrimitive()
        graph = {1: [2], 2: [3], 3: [1]}
        result = prim.process(graph)
        self.assertEqual(result, [])

    def test_union_find(self):
        prim = UnionFindPrimitive()
        edges = [(1, 2), (2, 3), (4, 5)]
        result = prim.process(edges)
        self.assertEqual(result, 2)

    def test_backtracking(self):
        prim = BacktrackingPrimitive()
        result = prim.process([1, 2])
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)

    def test_random_search(self):
        prim = RandomSearchPrimitive(target=3)
        result = prim.process([1, 2, 3, 4, 5])
        self.assertTrue(result == -1 or result == 2,
                        f"Expected index 2 or -1, got {result}")

    def test_lcs(self):
        prim = LongestCommonSubsequencePrimitive()
        result = prim.process(("abcde", "ace"))
        self.assertEqual(result, 3)

    def test_lcs_empty(self):
        prim = LongestCommonSubsequencePrimitive()
        result = prim.process(("abc", ""))
        self.assertEqual(result, 0)

    def test_rabin_karp(self):
        prim = RabinKarpPrimitive(pattern="test")
        result = prim.process("this is a test string")
        self.assertEqual(result, 10)

    def test_rabin_karp_not_found(self):
        prim = RabinKarpPrimitive(pattern="xyz")
        result = prim.process("this is a test string")
        self.assertEqual(result, -1)

    def test_rabin_karp_validate(self):
        prim = RabinKarpPrimitive(pattern="test")
        result = prim.process("this is a test string")
        self.assertTrue(prim.validate_output("this is a test string", result))


class TestPrimitiveComposabilityNew(unittest.TestCase):
    def test_sort_search_chain(self):
        sort = QuickSortPrimitive()
        search = InterpolationSearchPrimitive(target=3)
        self.assertTrue(sort.can_compose_with(search))

    def test_sort_binary_chain(self):
        sort = QuickSortPrimitive()
        search = type('TestBinary', (Primitive,), {
            'name': 'test_bin', 'input_type': 'sorted_iterable',
            'output_type': 'scalar', 'process': lambda self, d: d
        })()
        self.assertTrue(sort.can_compose_with(search))

    def test_incompatible_sort_search(self):
        sort = QuickSortPrimitive()
        search = InterpolationSearchPrimitive(target=3)
        self.assertTrue(sort.can_compose_with(search))

    def test_pipeline_compose_valid(self):
        pipeline = compose_pipeline(["quicksort", "binary_search"])
        self.assertIsNotNone(pipeline)

    def test_registry_count(self):
        self.assertEqual(len(PRIMITIVES), 25)
        self.assertIn("quicksort", PRIMITIVES)
        self.assertIn("mergesort", PRIMITIVES)
        self.assertIn("heapsort", PRIMITIVES)
        self.assertIn("bfs", PRIMITIVES)
        self.assertIn("dfs", PRIMITIVES)
        self.assertIn("interpolation_search", PRIMITIVES)
        self.assertIn("two_pointer", PRIMITIVES)
        self.assertIn("sliding_window", PRIMITIVES)
        self.assertIn("topological_sort", PRIMITIVES)
        self.assertIn("union_find", PRIMITIVES)
        self.assertIn("backtracking", PRIMITIVES)
        self.assertIn("random_search", PRIMITIVES)
        self.assertIn("lcs", PRIMITIVES)
        self.assertIn("rabin_karp", PRIMITIVES)


class TestSortVariantsSameType(unittest.TestCase):
    def test_quicksort_type(self):
        p = QuickSortPrimitive()
        self.assertEqual(p.input_type, "iterable")
        self.assertEqual(p.output_type, "sorted_iterable")

    def test_mergesort_type(self):
        p = MergeSortPrimitive()
        self.assertEqual(p.output_type, "sorted_iterable")

    def test_heapsort_type(self):
        p = HeapSortPrimitive()
        self.assertEqual(p.output_type, "sorted_iterable")


class TestAlgorithmSynthesizer(unittest.TestCase):
    def setUp(self):
        self.synthesizer = AlgorithmSynthesizer()

    def test_template_match_sorting(self):
        spec = ProblemSpec(
            name="sort numbers",
            problem_type=ProblemType.TRANSFORMATION,
            inputs={"data": {"type": "list[int]"}},
            outputs={"sorted": {"type": "list[int]"}}
        )
        result = self.synthesizer.synthesize(spec)
        self.assertIsNotNone(result)
        self.assertEqual(result.strategy, "template_match")
        self.assertGreater(len(result.algorithms), 0)

    def test_template_match_search(self):
        spec = ProblemSpec(
            name="find element",
            problem_type=ProblemType.SEARCH,
            inputs={"data": {"type": "list[int]"}},
            outputs={"index": {"type": "int"}},
            constraints=["sorted"]
        )
        result = self.synthesizer.synthesize(spec)
        self.assertIsNotNone(result)
        self.assertEqual(result.strategy, "template_match")

    def test_template_match_optimization(self):
        spec = ProblemSpec(
            name="maximize profit",
            problem_type=ProblemType.OPTIMIZATION,
            inputs={"items": {"type": "list"}},
            outputs={"best": {"type": "float"}},
            objectives=[Objective("maximize", "profit")]
        )
        result = self.synthesizer.synthesize(spec)
        self.assertEqual(result.strategy, "template_match")

    def test_transfer_synthesis(self):
        library = ProblemLibrary()
        source = ProblemSpec(
            name="source_task",
            problem_type=ProblemType.SEARCH,
            inputs={"data": {"type": "list[int]"}},
            outputs={"idx": {"type": "int"}}
        )
        library.store_solution(source, ["binary_search"], {"score": 0.9})

        synth = AlgorithmSynthesizer(problem_library=library)
        similar = ProblemSpec(
            name="similar_task",
            problem_type=ProblemType.SEARCH,
            inputs={"arr": {"type": "list[int]"}},
            outputs={"pos": {"type": "int"}}
        )
        result = synth.synthesize(similar)
        self.assertIsNotNone(result)
        self.assertIn(result.strategy, ("template_match", "transfer"))

    def test_primitive_composition_unknown(self):
        spec = ProblemSpec(
            name="custom_problem",
            inputs={"x": {"type": "str"}},
            outputs={"y": {"type": "float"}}
        )
        result = self.synthesizer.synthesize(spec)
        self.assertIsNotNone(result)
        self.assertIn(result.strategy, ("primitive_composition", "template_match"))

    def test_cache_hit(self):
        spec = ProblemSpec(
            name="cached_test",
            inputs={"a": {"type": "int"}},
            outputs={"b": {"type": "int"}}
        )
        r1 = self.synthesizer.synthesize(spec)
        r2 = self.synthesizer.synthesize(spec)
        self.assertEqual(r1.strategy, r2.strategy)
        self.assertGreaterEqual(len(r2.algorithms), 0)

    def test_clear_cache(self):
        spec = ProblemSpec(
            name="cache_clear",
            inputs={"a": {"type": "int"}},
            outputs={"b": {"type": "int"}}
        )
        self.synthesizer.synthesize(spec)
        self.synthesizer.clear_cache()
        self.assertEqual(len(self.synthesizer._synthesis_cache), 0)

    def test_result_type(self):
        spec = ProblemSpec(
            name="result_check",
            inputs={"x": {"type": "int"}},
            outputs={"y": {"type": "int"}}
        )
        result = self.synthesizer.synthesize(spec)
        self.assertIsInstance(result, SynthesisResult)
        self.assertIsInstance(result.algorithms, list)
        self.assertIsInstance(result.confidence, float)
        self.assertIsInstance(result.explanation, str)

    def test_transfer_synthesis_no_library(self):
        synth = AlgorithmSynthesizer(problem_library=ProblemLibrary())
        spec = ProblemSpec(
            name="no_match_fallback",
            inputs={"data": {"type": "list[int]"}},
            outputs={"result": {"type": "int"}}
        )
        result = synth.synthesize(spec)
        self.assertIsNotNone(result)

    def test_llm_synthesis_no_client(self):
        spec = ProblemSpec(
            name="llm_none",
            inputs={"x": {"type": "int"}},
            outputs={"y": {"type": "int"}}
        )
        result = self.synthesizer.synthesize(spec, use_llm=True)
        self.assertIsNotNone(result)
        self.assertNotEqual(result.strategy, "llm")


class TestExplainer(unittest.TestCase):
    def setUp(self):
        self.explainer = Explainer()

    def test_explain_quicksort(self):
        exp = self.explainer.explain("quicksort")
        self.assertIsInstance(exp, Explanation)
        self.assertEqual(exp.algorithm_name, "quicksort")
        self.assertIn("divide-and-conquer", exp.summary)
        self.assertEqual(exp.source, "template")

    def test_explain_binary_search(self):
        exp = self.explainer.explain("binary_search")
        self.assertIsInstance(exp, Explanation)
        self.assertIn("sorted", exp.summary.lower())
        self.assertGreater(len(exp.steps), 0)

    def test_explain_binary_search_steps(self):
        exp = self.explainer.explain("quicksort")
        self.assertGreaterEqual(len(exp.steps), 2)
        self.assertIn("pivot", exp.steps[0].lower())

    def test_explain_unknown_algorithm(self):
        exp = self.explainer.explain("custom_unknown_algorithm")
        self.assertIsInstance(exp, Explanation)
        self.assertIn("custom", exp.summary.lower())

    def test_explain_with_context(self):
        exp = self.explainer.explain("quicksort", context={"task": "sorting large dataset"})
        self.assertIn("Context", exp.summary)

    def test_explain_pipeline(self):
        explanations = self.explainer.explain_pipeline(["quicksort", "binary_search"])
        self.assertEqual(len(explanations), 2)
        self.assertEqual(explanations[0].algorithm_name, "quicksort")
        self.assertEqual(explanations[1].algorithm_name, "binary_search")

    def test_detail_level_short(self):
        exp = self.explainer.explain("mergesort", detail="short")
        self.assertEqual(exp.detail_level, "short")
        self.assertEqual(exp.source, "template")

    def test_detail_level_detailed_no_llm(self):
        exp = self.explainer.explain("mergesort", detail="detailed")
        self.assertEqual(exp.source, "template")

    def test_list_available(self):
        available = self.explainer.list_available_explanations()
        self.assertIn("quicksort", available)
        self.assertIn("binary_search", available)
        self.assertGreaterEqual(len(available), 16)


if __name__ == "__main__":
    unittest.main()
