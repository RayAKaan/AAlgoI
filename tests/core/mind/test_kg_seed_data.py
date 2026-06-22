from aalgoi.core.mind.kg_seed_data import (
    get_seed_algorithms,
    get_seed_principles,
    get_seed_problems,
)
from aalgoi.core.mind.knowledge_graph import AlgorithmNode, PrincipleNode, ProblemNode


class TestGetSeedAlgorithms:
    def test_returns_list(self):
        algos = get_seed_algorithms()
        assert isinstance(algos, list)

    def test_non_empty(self):
        algos = get_seed_algorithms()
        assert len(algos) > 0

    def test_all_are_algorithm_nodes(self):
        algos = get_seed_algorithms()
        for algo in algos:
            assert isinstance(algo, AlgorithmNode)

    def test_no_duplicate_names(self):
        algos = get_seed_algorithms()
        names = [a.name for a in algos]
        assert len(names) == len(set(names)), "Duplicate algorithm names found"

    def test_each_has_name_and_code(self):
        algos = get_seed_algorithms()
        for algo in algos:
            assert algo.name, f"Algorithm missing name"
            assert algo.code, f"Algorithm {algo.name} missing code"

    def test_each_has_time_complexity(self):
        algos = get_seed_algorithms()
        for algo in algos:
            assert algo.time_complexity, f"Algorithm {algo.name} missing time_complexity"

    def test_some_are_correctness_verified(self):
        algos = get_seed_algorithms()
        verified = [a for a in algos if a.correctness_verified]
        assert len(verified) > 0

    def test_includes_sorting_algorithms(self):
        algos = get_seed_algorithms()
        names = [a.name for a in algos]
        assert any("sort" in n.lower() for n in names)

    def test_includes_search_algorithms(self):
        algos = get_seed_algorithms()
        names = [a.name for a in algos]
        assert any("search" in n.lower() for n in names)

    def test_contains_tim_sort(self):
        algos = get_seed_algorithms()
        names = [a.name for a in algos]
        assert "tim_sort" in names

    def test_includes_graph_algorithms(self):
        algos = get_seed_algorithms()
        names = [a.name for a in algos]
        graph_names = {"bfs", "dfs", "dijkstra", "kruskal", "prim"}
        assert set(names) & graph_names

    def test_performance_history_is_float_list(self):
        algos = get_seed_algorithms()
        for algo in algos:
            assert isinstance(algo.performance_history, list)
            for val in algo.performance_history:
                assert isinstance(val, float)

    def test_principles_is_list_of_strings(self):
        algos = get_seed_algorithms()
        for algo in algos:
            assert isinstance(algo.principles, list)
            for p in algo.principles:
                assert isinstance(p, str)

    def test_best_for_is_list_of_strings(self):
        algos = get_seed_algorithms()
        for algo in algos:
            assert isinstance(algo.best_for, list)
            for bf in algo.best_for:
                assert isinstance(bf, str)

    def test_times_used_is_int(self):
        algos = get_seed_algorithms()
        for algo in algos:
            assert isinstance(algo.times_used, int)

    def test_times_succeeded_is_int(self):
        algos = get_seed_algorithms()
        for algo in algos:
            assert isinstance(algo.times_succeeded, int)

    def test_parent_algorithms_is_list(self):
        algos = get_seed_algorithms()
        for algo in algos:
            assert isinstance(algo.parent_algorithms, list)


class TestGetSeedPrinciples:
    def test_returns_list(self):
        principles = get_seed_principles()
        assert isinstance(principles, list)

    def test_all_are_principle_nodes(self):
        principles = get_seed_principles()
        for p in principles:
            assert isinstance(p, PrincipleNode)

    def test_non_empty(self):
        principles = get_seed_principles()
        assert len(principles) > 0

    def test_no_duplicate_names(self):
        principles = get_seed_principles()
        names = [p.name for p in principles]
        assert len(names) == len(set(names)), "Duplicate principle names found"


class TestGetSeedProblems:
    def test_returns_list(self):
        problems = get_seed_problems()
        assert isinstance(problems, list)

    def test_all_are_problem_nodes(self):
        problems = get_seed_problems()
        for p in problems:
            assert isinstance(p, ProblemNode)

    def test_non_empty(self):
        problems = get_seed_problems()
        assert len(problems) > 0

    def test_no_duplicate_signatures(self):
        problems = get_seed_problems()
        sigs = [p.signature for p in problems]
        assert len(sigs) == len(set(sigs)), "Duplicate problem signatures found"

    def test_each_has_signature_and_description(self):
        problems = get_seed_problems()
        for p in problems:
            assert p.signature, "Problem missing signature"
            assert p.description, f"Problem {p.signature} missing description"

    def test_each_has_domain(self):
        problems = get_seed_problems()
        for p in problems:
            assert p.domain, f"Problem {p.signature} missing domain"

    def test_each_has_best_algorithm(self):
        problems = get_seed_problems()
        for p in problems:
            assert p.best_algorithm, f"Problem {p.signature} missing best_algorithm"

    def test_difficulty_in_range(self):
        problems = get_seed_problems()
        for p in problems:
            assert 0.0 <= p.difficulty <= 1.0, (
                f"Problem {p.signature} difficulty {p.difficulty} out of range"
            )
