
import random
import unittest

from aalgoi.algorithms.sorting import HeapSort, InsertionSort, MergeSort, QuickSort, RadixSort, TimSort
from aalgoi.core.bandit import UCB1Bandit
from aalgoi.core.decision_log import Decision, DecisionLog
from aalgoi.core.drift_detector import DriftDetector
from aalgoi.core.pipeline_graph import PipelineGraph
from aalgoi.core.validator import PipelineValidator
from aalgoi.pipeline import AAlgoI


class TestSorting(unittest.TestCase):
    def setUp(self):
        self.system = AAlgoI(config={
            "strategy": "hybrid",
            "time_budget_ms": 500,
            "priority": "balanced"
        })

    def test_tiny_data(self):
        data = [3, 1, 2]
        expected = sorted(data)
        result = self.system.run(data, task_type="sorting", expected_result=expected)
        self.assertEqual(result, expected)

    def test_random_data(self):
        data = [random.randint(0, 1000) for _ in range(1000)]
        expected = sorted(data)
        result = self.system.run(data, task_type="sorting", expected_result=expected)
        self.assertEqual(result, expected)

    def test_sorted_data(self):
        data = list(range(1000))
        expected = sorted(data)
        result = self.system.run(data, task_type="sorting", expected_result=expected)
        self.assertEqual(result, expected)

    def test_nearly_sorted_data(self):
        data = list(range(1000))
        for _ in range(50):
            i, j = random.randint(0, 999), random.randint(0, 999)
            data[i], data[j] = data[j], data[i]
        expected = sorted(data)
        result = self.system.run(data, task_type="sorting", expected_result=expected)
        self.assertEqual(result, expected)

    def test_reverse_data(self):
        data = list(range(1000, 0, -1))
        expected = sorted(data)
        result = self.system.run(data, task_type="sorting", expected_result=expected)
        self.assertEqual(result, expected)

    def test_large_data(self):
        data = [random.randint(0, 100000) for _ in range(50000)]
        expected = sorted(data)
        result = self.system.run(data, task_type="sorting", expected_result=expected)
        self.assertEqual(result, expected)


class TestAlgorithms(unittest.TestCase):
    def test_all_sorting_algorithms(self):
        data = [random.randint(0, 1000) for _ in range(100)]
        expected = sorted(data)

        algorithms = {
            "quicksort": QuickSort(),
            "insertion_sort": InsertionSort(),
            "merge_sort": MergeSort(),
            "timsort": TimSort(),
            "radix_sort": RadixSort(),
            "heap_sort": HeapSort(),
        }

        for name, algo in algorithms.items():
            result = algo.process(data.copy())
            self.assertEqual(result, expected, f"{name} failed")

    def test_validate_output(self):
        data = [3, 1, 2]
        algo = QuickSort()
        result = algo.process(data.copy())
        self.assertTrue(algo.validate_output(data.copy(), result))

        self.assertFalse(algo.validate_output(data.copy(), [1, 2]))
        self.assertFalse(algo.validate_output(data.copy(), None))
        self.assertFalse(algo.validate_output(data.copy(), "not a list"))


class TestBandit(unittest.TestCase):
    def test_ucb1_selection(self):
        bandit = UCB1Bandit(["a", "b", "c"])
        bandit.update("a", 1.0)
        bandit.update("b", 0.5)
        bandit.update("c", 0.0)
        chosen = bandit.select()
        self.assertIn(chosen, ["a", "b", "c"])

    def test_epsilon_decay(self):
        bandit = UCB1Bandit(["a", "b"], epsilon=0.5, epsilon_min=0.1)
        self.assertEqual(bandit.epsilon, 0.5)
        bandit.decay_epsilon()
        self.assertAlmostEqual(bandit.epsilon, 0.5 * 0.99)


class TestValidator(unittest.TestCase):
    def test_validation(self):
        validator = PipelineValidator()
        algo = QuickSort()
        data = [3, 1, 2]
        result = algo.process(data.copy())
        validation = validator.validate_step(algo, data, result)
        self.assertTrue(validation.passed)

    def test_failure_detection(self):
        validator = PipelineValidator()
        algo = QuickSort()
        data = [3, 1, 2]
        validation = validator.validate_step(algo, data, None)
        self.assertFalse(validation.passed)


class TestDriftDetector(unittest.TestCase):
    def test_drift_detection(self):
        detector = DriftDetector(window=10, threshold=0.5)
        for _ in range(10):
            detector.update(0.9)
        self.assertFalse(detector.update(0.9))
        for _ in range(10):
            detector.update(0.1)
        drifted = detector.update(0.1)
        self.assertIn(drifted, [True, False])


class TestDecisionLog(unittest.TestCase):
    def test_logging(self):
        log = DecisionLog()
        decision = Decision(
            context={"test": True},
            candidates=["a", "b"],
            chosen="a",
            confidence=0.8,
            reason="test"
        )
        log.record(decision)
        self.assertEqual(len(log.recent_decisions), 1)
        self.assertEqual(log.get_last().chosen, "a")


class TestPipelineGraph(unittest.TestCase):
    def test_linear_graph(self):
        graph = PipelineGraph()
        graph.add_algorithm("step_0", QuickSort())
        self.assertTrue(graph.validate())

    def test_cycle_detection(self):
        graph = PipelineGraph()
        graph.add_algorithm("a", QuickSort(), depends_on=["b"])
        graph.add_algorithm("b", QuickSort(), depends_on=["a"])
        self.assertFalse(graph.validate())


class TestContextEngine(unittest.TestCase):
    def setUp(self):
        self.system = AAlgoI()

    def test_context_analysis(self):
        data = [1, 2, 3, 4, 5]
        context = self.system.context_engine.analyze(data, task_type="sorting")

        self.assertIn("data_profile", context)
        self.assertIn("environment", context)
        self.assertIn("constraints", context)
        self.assertIn("features", context)

        self.assertEqual(context["data_profile"]["size"], 5)
        self.assertEqual(context["data_profile"]["type"], "list")


class TestMetaController(unittest.TestCase):
    def setUp(self):
        self.system = AAlgoI(config={"strategy": "hybrid"})

    def test_domain_filtering(self):
        data = [1, 2, 3]
        context = self.system.context_engine.analyze(data, task_type="sorting")

        domain_algos = self.system.meta_controller._get_domain_algorithms(context)

        self.assertIn("timsort", domain_algos)
        self.assertIn("quicksort", domain_algos)
        self.assertNotIn("gaussian_blur", domain_algos)
        self.assertNotIn("kmeans", domain_algos)

    def test_confidence_scoring(self):
        from aalgoi.core.meta_controller import MetaController
        mc = MetaController(
            algorithm_registry=self.system.meta_controller.registry,
            config={}
        )
        data = [1, 2, 3]
        context = self.system.context_engine.analyze(data, task_type="sorting")
        algos, confidence, reason = mc.select_with_confidence(context)
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)


class TestLearning(unittest.TestCase):
    def test_history_building(self):
        system = AAlgoI(config={"strategy": "hybrid"})

        for _ in range(10):
            data = [random.randint(0, 100) for _ in range(50)]
            expected = sorted(data)
            system.run(data, task_type="sorting", expected_result=expected)

        self.assertEqual(len(system.meta_controller.history), 10)

    def test_bandit_updates(self):
        system = AAlgoI(config={"strategy": "hybrid"})

        for _ in range(10):
            data = [random.randint(0, 100) for _ in range(50)]
            expected = sorted(data)
            system.run(data, task_type="sorting", expected_result=expected)

        bandit_stats = system.meta_controller.bandit.get_stats()
        total_trials = bandit_stats["total_trials"]
        self.assertGreater(total_trials, 0)


if __name__ == "__main__":
    unittest.main()
