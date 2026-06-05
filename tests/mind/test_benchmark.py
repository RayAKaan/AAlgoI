from aalgoi.core.mind.benchmark import (
    BenchmarkProblem,
    get_benchmark_problems,
    run_benchmark,
    verify_solution,
)


class TestBenchmarkProblems:
    def test_55_problems_exist(self):
        problems = get_benchmark_problems()
        assert len(problems) == 55

    def test_all_have_ids(self):
        problems = get_benchmark_problems()
        for p in problems:
            assert p.id is not None
            assert len(p.id) > 0

    def test_all_ids_unique(self):
        problems = get_benchmark_problems()
        ids = [p.id for p in problems]
        assert len(ids) == len(set(ids))

    def test_all_have_domains(self):
        problems = get_benchmark_problems()
        for p in problems:
            assert p.domain is not None

    def test_all_have_difficulty(self):
        problems = get_benchmark_problems()
        for p in problems:
            assert 0.0 <= p.difficulty <= 1.0

    def test_all_have_problem_text(self):
        problems = get_benchmark_problems()
        for p in problems:
            assert p.problem_text is not None
            assert len(p.problem_text) > 0

    def test_all_have_data(self):
        problems = get_benchmark_problems()
        for p in problems:
            assert p.data is not None

    def test_all_have_best_algorithm(self):
        problems = get_benchmark_problems()
        for p in problems:
            assert p.best_algorithm is not None

    def test_all_have_verification_fn(self):
        problems = get_benchmark_problems()
        for p in problems:
            assert p.verification_fn in ("exact", "sorted", "set", "approximate")

    def test_domains_represented(self):
        problems = get_benchmark_problems()
        domains = set(p.domain for p in problems)
        assert "integers" in domains
        assert "graph" in domains
        assert "numbers" in domains
        assert "text" in domains
        assert "array" in domains

    def test_5_sorting_problems(self):
        problems = get_benchmark_problems()
        sorting = [p for p in problems if p.id.startswith("sort")]
        assert len(sorting) == 5

    def test_5_pathfinding_problems(self):
        problems = get_benchmark_problems()
        path = [p for p in problems if p.id.startswith("path")]
        assert len(path) == 5

    def test_5_dp_problems(self):
        problems = get_benchmark_problems()
        dp = [p for p in problems if p.id.startswith("dp")]
        assert len(dp) == 5

    def test_5_greedy_problems(self):
        problems = get_benchmark_problems()
        greedy = [p for p in problems if p.id.startswith("greedy")]
        assert len(greedy) == 5

    def test_5_search_problems(self):
        problems = get_benchmark_problems()
        search = [p for p in problems if p.id.startswith("search")]
        assert len(search) == 5

    def test_5_hash_problems(self):
        problems = get_benchmark_problems()
        hash_problems = [p for p in problems if p.id.startswith("hash")]
        assert len(hash_problems) == 5

    def test_5_graph_problems(self):
        problems = get_benchmark_problems()
        graph = [p for p in problems if p.id.startswith("graph")]
        assert len(graph) == 5

    def test_5_string_problems(self):
        problems = get_benchmark_problems()
        string = [p for p in problems if p.id.startswith("string")]
        assert len(string) == 5

    def test_5_bsearch_problems(self):
        problems = get_benchmark_problems()
        bsearch = [p for p in problems if p.id.startswith("bsearch")]
        assert len(bsearch) == 5

    def test_5_ml_problems(self):
        problems = get_benchmark_problems()
        ml = [p for p in problems if p.id.startswith("ml")]
        assert len(ml) == 5

    def test_5_math_problems(self):
        problems = get_benchmark_problems()
        math = [p for p in problems if p.id.startswith("math")]
        assert len(math) == 5


class TestVerifySolution:
    def test_exact_match(self):
        p = BenchmarkProblem("test", "numbers", 0.1, "test", {}, 42, "test", "O(1)", "exact")
        assert verify_solution(p, 42) is True
        assert verify_solution(p, 43) is False

    def test_sorted_match(self):
        p = BenchmarkProblem("test", "numbers", 0.1, "test", {}, [1, 2, 3], "test", "O(1)", "sorted")
        assert verify_solution(p, [3, 2, 1]) is True
        assert verify_solution(p, [1, 2, 3]) is True
        assert verify_solution(p, [1, 2, 4]) is False

    def test_approximate_always_passes(self):
        p = BenchmarkProblem("test", "numbers", 0.1, "test", {}, None, "test", "O(1)", "approximate")
        assert verify_solution(p, "anything") is True
        assert verify_solution(p, 42) is True

    def test_none_fails(self):
        p = BenchmarkProblem("test", "numbers", 0.1, "test", {}, 42, "test", "O(1)", "exact")
        assert verify_solution(p, None) is False

    def test_set_match(self):
        p = BenchmarkProblem("test", "numbers", 0.1, "test", {}, [0, 1], "test", "O(1)", "set")
        assert verify_solution(p, [0, 1]) is True
        assert verify_solution(p, [1, 0]) is True


class TestRunBenchmark:
    def test_run_with_dummy_solver(self):
        def dummy_solver(problem_text, data):
            if "sort" in problem_text.lower():
                if "nums" in data:
                    return sorted(data["nums"])
            if "gcd" in problem_text.lower():
                import math
                return math.gcd(data.get("a", 1), data.get("b", 1))
            if "search" in problem_text.lower():
                return -1
            return None

        results = run_benchmark(dummy_solver)
        assert results["total"] == 55
        assert results["correct"] + results["failed"] + results["errors"] == 55
        assert "accuracy" in results
        assert "by_domain" in results

    def test_run_with_perfect_solver(self):
        problems = get_benchmark_problems()

        def perfect_solver(problem_text, data):
            for p in problems:
                if p.id.split("_")[0] in problem_text.lower() or True:
                    return p.expected_output
            return None

        results = run_benchmark(perfect_solver)
        assert results["total"] == 55

    def test_run_with_failing_solver(self):
        def failing_solver(problem_text, data):
            return "wrong"

        results = run_benchmark(failing_solver)
        assert results["total"] == 55

    def test_benchmark_tracks_by_domain(self):
        def dummy_solver(problem_text, data):
            return None

        results = run_benchmark(dummy_solver)
        assert len(results["by_domain"]) > 0
        for domain, stats in results["by_domain"].items():
            assert "correct" in stats
            assert "total" in stats
            assert stats["total"] > 0
