from aalgoi.algorithms.registry import get_registry
from aalgoi.types import ProblemSpec, ProblemTask, Domain


from aalgoi.algorithms.registry import get_registry
from aalgoi.types import ProblemSpec, ProblemTask, Domain


def test_bfs_integer_nodes():
    registry = get_registry()
    algo = registry.create("bfs")
    spec = ProblemSpec(
        id="bfs_int",
        task=ProblemTask.BFS,
        domain=Domain.GRAPH,
        inputs={"graph": {0: [1, 2], 1: [3], 2: [3], 3: []}, "start": 0},
    )
    result = algo.run(spec)
    assert result == [0, 1, 2, 3]


def test_bfs_integer_nodes_no_start():
    registry = get_registry()
    algo = registry.create("bfs")
    spec = ProblemSpec(
        id="bfs_int_nostart",
        task=ProblemTask.BFS,
        domain=Domain.GRAPH,
        inputs={"graph": {5: [3], 3: []}},
    )
    result = algo.run(spec)
    assert result == [5, 3]


def test_dfs_integer_nodes():
    registry = get_registry()
    algo = registry.create("dfs")
    spec = ProblemSpec(
        id="dfs_int",
        task=ProblemTask.DFS,
        domain=Domain.GRAPH,
        inputs={"graph": {0: [1, 2], 1: [3], 2: [], 3: []}, "start": 0},
    )
    result = algo.run(spec)
    assert result == [0, 1, 3, 2]


def test_shortest_path_integer_nodes():
    registry = get_registry()
    algo = registry.create("shortest_path_unweighted")
    spec = ProblemSpec(
        id="sp_int",
        task=ProblemTask.SHORTEST_PATH_UNWEIGHTED,
        domain=Domain.GRAPH,
        inputs={"graph": {0: [1], 1: [2], 2: [3], 3: []}, "start": 0, "end": 3},
    )
    result = algo.run(spec)
    assert result == [0, 1, 2, 3]


def test_dijkstra_integer_nodes():
    registry = get_registry()
    algo = registry.create("dijkstra")
    spec = ProblemSpec(
        id="dijk_int",
        task=ProblemTask.SHORTEST_PATH_WEIGHTED,
        domain=Domain.GRAPH,
        inputs={"graph": {0: {1: 5}, 1: {2: 3}, 2: {3: 1}, 3: {}}, "start": 0, "end": 3},
    )
    result = algo.run(spec)
    assert result["path"] == [0, 1, 2, 3]


def test_bellman_ford_integer_nodes():
    registry = get_registry()
    algo = registry.create("bellman_ford")
    spec = ProblemSpec(
        id="bf_int",
        task=ProblemTask.SHORTEST_PATH_NEGATIVE,
        domain=Domain.GRAPH,
        inputs={"graph": {0: {1: 4}, 1: {2: -2}, 2: {3: 3}, 3: {}}, "start": 0, "end": 3},
    )
    result = algo.run(spec)
    assert result["path"] == [0, 1, 2, 3]


def test_connected_components_integer_nodes():
    registry = get_registry()
    algo = registry.create("connected_components")
    spec = ProblemSpec(
        id="cc_int",
        task=ProblemTask.CONNECTED_COMPONENTS,
        domain=Domain.GRAPH,
        inputs={"graph": {0: [1], 2: [3]}},
    )
    result = algo.run(spec)
    assert len(result) == 2


def test_edmonds_karp_integer_nodes():
    registry = get_registry()
    algo = registry.create("edmonds_karp")
    spec = ProblemSpec(
        id="ek_int",
        task=ProblemTask.MAX_FLOW,
        domain=Domain.GRAPH,
        inputs={"graph": {0: {1: 10, 2: 5}, 1: {2: 3, 3: 8}, 2: {3: 6}}, "source": 0, "sink": 3},
    )
    result = algo.run(spec)
    assert result["flow_value"] == 14.0
