from aalgoi.algorithms.optimization import GreedyKnapsack, SimulatedAnnealing

ITEMS = [
    {"value": 60, "weight": 10},
    {"value": 100, "weight": 20},
    {"value": 120, "weight": 30},
]


def test_greedy_knapsack_selects_best_ratio():
    g = GreedyKnapsack()
    result = g.process({"items": ITEMS, "capacity": 50})
    assert result["value"] == 160.0
    assert result["weight"] == 30.0
    assert result["selected"] == [0, 1]


def test_greedy_knapsack_respects_capacity():
    g = GreedyKnapsack()
    result = g.process({"items": ITEMS, "capacity": 15})
    assert result["value"] == 60.0
    assert result["weight"] == 10.0


def test_greedy_knapsack_validate_output():
    g = GreedyKnapsack()
    assert g.validate_output(None, {"selected": [0], "value": 10}) is True
    assert g.validate_output(None, {}) is False
    assert g.validate_output(None, None) is False


def test_simulated_annealing_returns_valid_result():
    sa = SimulatedAnnealing()
    result = sa.process({"items": ITEMS, "capacity": 50, "iterations": 200})
    assert "selected" in result
    assert "value" in result
    assert "weight" in result
    assert result["value"] > 0
    assert result["weight"] <= 50


def test_simulated_annealing_respects_capacity():
    sa = SimulatedAnnealing()
    result = sa.process({"items": ITEMS, "capacity": 15, "iterations": 200})
    assert result["weight"] <= 15


def test_simulated_annealing_validate_output():
    sa = SimulatedAnnealing()
    assert sa.validate_output(None, {"selected": [0], "value": 10}) is True
    assert sa.validate_output(None, None) is False


def test_greedy_knapsack_tags():
    g = GreedyKnapsack()
    assert "optimization" in g.tags
    assert "knapsack" in g.tags


def test_simulated_annealing_tags():
    sa = SimulatedAnnealing()
    assert "metaheuristic" in sa.tags
    assert "combinatorial" in sa.tags


def test_optimization_auto_detect_from_tuple():
    from aalgoi.core.problem_spec import ProblemSpec, ProblemType
    data = ProblemSpec._infer_from_data_shape((ITEMS, 50))
    assert data == ProblemType.OPTIMIZATION


def test_optimization_auto_detect_from_dict():
    from aalgoi.core.problem_spec import ProblemSpec, ProblemType
    data = ProblemSpec._infer_from_data_shape({"items": ITEMS, "capacity": 50})
    assert data == ProblemType.OPTIMIZATION
