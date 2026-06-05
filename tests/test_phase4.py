import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aalgoi.core.problem_spec import ProblemType
from interface.nl_parser import PATTERNS, extract_data_from_description, parse_description, parse_solve_input


class TestNLParseSorting(unittest.TestCase):
    def test_sort_numbers(self):
        spec = parse_description("sort these numbers in ascending order")
        self.assertEqual(spec.problem_type, ProblemType.TRANSFORMATION)
        self.assertIn("data", spec.inputs)

    def test_sort_description_name(self):
        desc = "sort this list of integers"
        spec = parse_description(desc)
        self.assertIn("sort", spec.name)

    def test_sort_extract_data(self):
        data = extract_data_from_description("sort numbers: 5, 3, 1, 4, 2")
        self.assertEqual(data, [5, 3, 1, 4, 2])


class TestNLParseSearch(unittest.TestCase):
    def test_find_element(self):
        spec = parse_description("find the number 42 in the list")
        self.assertEqual(spec.problem_type, ProblemType.SEARCH)

    def test_search_output_type(self):
        spec = parse_description("locate the target value in array")
        self.assertEqual(spec.outputs.get("result", {}).get("type"), "int")


class TestNLParseOptimization(unittest.TestCase):
    def test_maximize(self):
        spec = parse_description("maximize profit given constraints")
        self.assertEqual(spec.problem_type, ProblemType.OPTIMIZATION)

    def test_minimize_detection(self):
        spec = parse_description("minimize the total cost of the project")
        self.assertEqual(spec.problem_type, ProblemType.OPTIMIZATION)

    def test_direction_detected(self):
        spec = parse_description("maximize profit given constraints")
        self.assertGreater(len(spec.objectives), 0)


class TestNLParseRouting(unittest.TestCase):
    def test_routing_detection(self):
        spec = parse_description("find the shortest path between two cities on a map")
        self.assertEqual(spec.problem_type, ProblemType.ROUTING)
        self.assertGreater(len(spec.constraints), 0)


class TestNLParseScheduling(unittest.TestCase):
    def test_scheduling_detection(self):
        spec = parse_description("schedule these tasks to meet deadlines")
        self.assertEqual(spec.problem_type, ProblemType.SCHEDULING)


class TestNLParseClassification(unittest.TestCase):
    def test_classify_detection(self):
        spec = parse_description("classify these items into categories")
        self.assertEqual(spec.problem_type, ProblemType.CLASSIFICATION)

    def test_classify_output(self):
        spec = parse_description("classify these items into categories")
        self.assertEqual(spec.outputs.get("result", {}).get("type"), "int")


class TestNLParseInputInference(unittest.TestCase):
    def test_graph_input(self):
        spec = parse_description("find the shortest path in a graph with weighted edges")
        self.assertIn("graph", spec.inputs)

    def test_string_input(self):
        spec = parse_description("search for a pattern in a string")
        self.assertIn("text", spec.inputs)

    def test_fallback_input(self):
        spec = parse_description("do something with the data")
        self.assertIn("data", spec.inputs)


class TestNLParseSolveInput(unittest.TestCase):
    def test_parse_solve_input(self):
        spec, data = parse_solve_input("sort numbers: 10, 5, 2, 7")
        self.assertIsNotNone(spec)
        self.assertEqual(data, [10, 5, 2, 7])

    def test_parse_solve_no_numbers(self):
        spec, data = parse_solve_input("sort this list")
        self.assertIsNotNone(spec)
        self.assertIsNone(data)


class TestNLParseAllPatterns(unittest.TestCase):
    def test_all_patterns_have_required_keys(self):
        for p in PATTERNS:
            self.assertIn("name", p)
            self.assertIn("keywords", p)
            self.assertIn("problem_type", p)
            self.assertIn("output_type", p)
            self.assertGreater(len(p["keywords"]), 0)


class TestAPICreation(unittest.TestCase):
    def test_app_creation(self):
        try:
            from interface.api import create_app
            app = create_app()
            self.assertIsNotNone(app)
        except ImportError:
            self.skipTest("FastAPI not available")

    def test_app_solve_endpoint_exists(self):
        try:
            from interface.api import create_app
            app = create_app()
            routes = [r.path for r in app.routes]
            self.assertIn("/solve", routes)
        except ImportError:
            self.skipTest("FastAPI not available")

    def test_app_health_endpoint(self):
        try:
            from interface.api import create_app
            app = create_app()
            routes = [r.path for r in app.routes]
            self.assertIn("/health", routes)
        except ImportError:
            self.skipTest("FastAPI not available")


class TestCLICommands(unittest.TestCase):
    def test_cli_imports(self):
        try:
            from interface.cli import main
            self.assertIsNotNone(main)
        except ImportError:
            self.skipTest("CLI module issue")

    def test_subcommands_exist(self):
        try:
            from interface.cli import main
            commands = list(main.commands.keys())
            self.assertIn("solve", commands)
            self.assertIn("explain", commands)
            self.assertIn("stats", commands)
            self.assertIn("ml", commands)
        except ImportError:
            self.skipTest("CLI module issue")


class TestWebUIAvailability(unittest.TestCase):
    def test_import_gradio(self):
        try:
            from interface.web_ui import create_interface, launch
            self.assertTrue(True)
        except ImportError:
            self.skipTest("Gradio not available")


if __name__ == "__main__":
    unittest.main()
