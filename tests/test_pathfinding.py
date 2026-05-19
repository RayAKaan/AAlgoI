import pytest
from algorithms.pathfinding import Dijkstra, AStar, BFSPathfinder


SIMPLE_GRAPH = {"A": {"B": 1, "C": 4}, "B": {"C": 2, "D": 5}, "C": {"D": 1}, "D": {}}


def test_dijkstra_finds_shortest_path():
    d = Dijkstra()
    result = d.process({"graph": SIMPLE_GRAPH, "start": "A", "end": "D"})
    assert result == ["A", "B", "C", "D"]


def test_dijkstra_no_end_returns_all_visited():
    d = Dijkstra()
    result = d.process({"graph": SIMPLE_GRAPH, "start": "A"})
    assert isinstance(result, set)
    assert result == {"A", "B", "C", "D"}


def test_dijkstra_no_path_returns_empty():
    d = Dijkstra()
    result = d.process({"graph": {"A": {}, "B": {}}, "start": "A", "end": "B"})
    assert result == []


def test_dijkstra_validate_output():
    d = Dijkstra()
    assert d.validate_output(None, ["A", "B"]) is True
    assert d.validate_output(None, {"A"}) is True
    assert d.validate_output(None, "invalid") is False


def test_astar_finds_path():
    a = AStar()
    result = a.process({"graph": SIMPLE_GRAPH, "start": "A", "end": "D"})
    assert result == ["A", "B", "C", "D"]


def test_astar_validate_output():
    a = AStar()
    assert a.validate_output(None, ["A", "B"]) is True
    assert a.validate_output(None, "invalid") is False


def test_bfs_pathfinder_finds_path():
    b = BFSPathfinder()
    result = b.process({"graph": SIMPLE_GRAPH, "start": "A", "end": "D"})
    assert result == ["A", "C", "D"] or result == ["A", "B", "D"]


def test_bfs_pathfinder_no_path_returns_empty():
    b = BFSPathfinder()
    result = b.process({"graph": {"A": {}, "B": {}}, "start": "A", "end": "B"})
    assert result == []


def test_bfs_validate_output():
    b = BFSPathfinder()
    assert b.validate_output(None, ["A"]) is True
    assert b.validate_output(None, None) is False


def test_dijkstra_tags():
    d = Dijkstra()
    assert "pathfinding" in d.tags
    assert "weighted" in d.tags


def test_astar_tags():
    a = AStar()
    assert "heuristic" in a.tags


def test_bfs_tags():
    b = BFSPathfinder()
    assert "unweighted" in b.tags


def test_data_auto_wrap_from_tuple():
    from core.problem_spec import ProblemSpec, ProblemType
    ps = ProblemSpec(name="test", problem_type=ProblemType.PATHFINDING)
    data = ps._infer_from_data_shape((SIMPLE_GRAPH, "A", "D"))
    assert data == ProblemType.PATHFINDING


def test_data_auto_wrap_from_dict_key():
    from core.problem_spec import ProblemSpec, ProblemType
    data = ProblemSpec._infer_from_data_shape({"graph": SIMPLE_GRAPH, "start": "A", "end": "D"})
    assert data == ProblemType.PATHFINDING
