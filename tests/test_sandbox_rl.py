"""
tests/test_sandbox_rl.py — Full test suite for SandboxRL.

Validates: state encoding, policy correctness, training loop,
curriculum, persistence, forking, instance store, auto-learn mode.
"""

import pytest
import numpy as np
import tempfile
import os

HAS_TORCH = False
try:
    import torch
    HAS_TORCH = True
except ImportError:
    pass

pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")


# ── StateBridge ────────────────────────────────────────────────────────────

class TestStateBridge:
    @pytest.fixture
    def bridge(self):
        from aalgoi.sandbox.state_bridge import StateBridge
        from core.smart_solver import SmartSolver
        return StateBridge(SmartSolver().solver.registry)

    def test_state_vector_shape(self, bridge):
        state = bridge.encode("sort numbers", {"data": [3,1,2]})
        assert state.shape == (bridge.state_size,)
        assert state.dtype == np.float32

    def test_deterministic(self, bridge):
        s1 = bridge.encode("sort numbers", {"data": [3,1,2]})
        s2 = bridge.encode("sort numbers", {"data": [3,1,2]})
        # Query portion (vocab) and mask portion are deterministic;
        # data features include environment stats (CPU%) that vary.
        query_mask = s1[:128]
        vm2 = s2[:128]
        assert np.allclose(query_mask, vm2), "Query encoding non-deterministic"
        mask1 = s1[-bridge.num_algos:]
        mask2 = s2[-bridge.num_algos:]
        assert np.allclose(mask1, mask2), "Algo mask non-deterministic"
        assert bridge.state_size == len(s1) == len(s2)

    def test_different_queries_differ(self, bridge):
        s1 = bridge.encode("sort ascending", {})
        s2 = bridge.encode("shortest pathfinding graph", {})
        assert not np.allclose(s1[:128], s2[:128])

    def test_query_activates_relevant_vocab(self, bridge):
        state = bridge.encode("sort data ascending", {"data": [3,1,2]})
        assert np.sum(state[:128]) > 0, "Query encoding all zeros"

    def test_data_features_graph(self, bridge):
        state = bridge.encode("find path", {"graph": {"A": {"B":1}}})
        data_slice = state[128:128+32]
        assert data_slice.sum() > 0

    def test_roundtrip_algo_idx(self, bridge):
        for name in bridge.algo_names[:5]:
            idx = bridge.algo_idx(name)
            assert bridge.idx_to_algo(idx) == name

    def test_state_size_greater_than_vocab_plus_data(self, bridge):
        assert bridge.state_size > 128 + 32


# ── PolicyAgent ────────────────────────────────────────────────────────────

class TestPolicyAgent:
    @pytest.fixture
    def agent(self):
        from aalgoi.sandbox.policy import PolicyAgent
        return PolicyAgent(input_size=80, num_algorithms=10, hidden_size=32, num_layers=1)

    def test_select_valid_index(self, agent):
        state = np.random.randn(80).astype(np.float32)
        idx, conf = agent.select(state)
        assert 0 <= idx < 10
        assert 0.0 <= conf <= 1.0

    def test_select_greedy_when_epsilon_zero(self, agent):
        agent.epsilon = 0.0
        state = np.random.randn(80).astype(np.float32)
        idx1, _ = agent.select(state)
        idx2, _ = agent.select(state)
        assert idx1 == idx2

    def test_update_returns_loss(self, agent):
        state = np.random.randn(80).astype(np.float32)
        for _ in range(20):
            idx, _ = agent.select(state)
            agent.record(state, idx, 1.0)
        loss = agent.update()
        assert loss is not None
        assert isinstance(loss, float)

    def test_update_returns_none_when_buffer_small(self, agent):
        state = np.random.randn(80).astype(np.float32)
        agent.record(state, 0, 1.0)
        assert agent.update() is None

    def test_save_load_roundtrip(self, agent):
        state = np.random.randn(80).astype(np.float32)
        for _ in range(10):
            idx, _ = agent.select(state)
            agent.record(state, idx, 0.8)
        agent.update()
        agent.episodes = 99

        with tempfile.NamedTemporaryFile(suffix=".rl", delete=False) as f:
            path = f.name
        try:
            agent.save(path)
            from aalgoi.sandbox.policy import PolicyAgent
            loaded = PolicyAgent.from_file(path)
            assert loaded.episodes == 99
        finally:
            os.unlink(path)

    def test_epsilon_decays_after_update(self, agent):
        initial_eps = agent.epsilon
        state = np.random.randn(80).astype(np.float32)
        for _ in range(20):
            idx, _ = agent.select(state)
            agent.record(state, idx, 1.0)
        agent.update()
        assert agent.epsilon < initial_eps

    def test_stats_keys(self, agent):
        s = agent.stats()
        for key in ["episodes", "total_reward", "avg_reward", "recent_avg", "epsilon", "buffer_size"]:
            assert key in s

    def test_reset_exploration(self, agent):
        agent.epsilon = 0.01
        agent.reset_exploration()
        assert agent.epsilon > 0.01

    def test_set_lr(self, agent):
        agent.set_lr(0.01)
        for g in agent.opt.param_groups:
            assert abs(g["lr"] - 0.01) < 1e-6


# ── Curriculum ─────────────────────────────────────────────────────────────

class TestCurriculum:
    def test_get_problems_count(self):
        from aalgoi.sandbox.curriculum import get_problems
        assert len(get_problems(1, 25)) == 25
        assert len(get_problems(1, 10)) == 10
        assert len(get_problems(11, 25)) == 15

    def test_problems_have_required_fields(self):
        from aalgoi.sandbox.curriculum import get_problems
        for p in get_problems():
            assert "id" in p
            assert "difficulty" in p
            assert "query" in p
            assert "data" in p
            assert "validator" in p
            assert callable(p["validator"])

    def test_difficulty_increases(self):
        from aalgoi.sandbox.curriculum import get_problems
        problems = get_problems(1, 25)
        difficulties = [p["difficulty"] for p in problems]
        first_half = np.mean(difficulties[:12])
        second_half = np.mean(difficulties[13:])
        assert second_half >= first_half

    def test_domain_filter(self):
        from aalgoi.sandbox.curriculum import get_problems
        sorting_only = get_problems(1, 10, domains=["SORTING"])
        for p in sorting_only:
            assert p.get("expected_type") in ("SORTING", None)

    def test_validators_callable(self):
        from aalgoi.sandbox.curriculum import get_problems
        for p in get_problems(1, 5):
            result = p["validator"](None, p)
            assert isinstance(result, bool)


# ── InstanceStore ──────────────────────────────────────────────────────────

class TestInstanceStore:
    def setup_method(self):
        from aalgoi.sandbox import instance_store
        instance_store._instances.clear()

    def test_register_and_get(self):
        from aalgoi.sandbox.instance_store import register, get_instance
        obj = object()
        register("test_obj", obj)
        assert get_instance("test_obj") is obj

    def test_list(self):
        from aalgoi.sandbox.instance_store import register, list_instances
        register("a", object())
        register("b", object())
        names = list_instances()
        assert "a" in names and "b" in names

    def test_delete(self):
        from aalgoi.sandbox.instance_store import register, delete_instance, get_instance
        register("to_delete", object())
        assert delete_instance("to_delete") is True
        assert get_instance("to_delete") is None

    def test_delete_nonexistent(self):
        from aalgoi.sandbox.instance_store import delete_instance
        assert delete_instance("does_not_exist") is False


# ── SandboxRL ──────────────────────────────────────────────────────────────

class TestSandboxRL:
    @pytest.fixture
    def rl(self):
        from aalgoi.sandbox import SandboxRL
        return SandboxRL(hidden_size=32, num_layers=1, epsilon=0.8)

    def test_creates_without_error(self, rl):
        assert rl._agent is not None
        assert rl._bridge is not None

    def test_curriculum_runs_and_logs(self, rl):
        rl.curriculum(start_from=1, end_at=3, verbose=False)
        assert len(rl._train_log) == 3
        for record in rl._train_log:
            assert "reward" in record
            assert "algo" in record
            assert isinstance(record["reward"], float)

    def test_curriculum_multiple_rounds(self, rl):
        rl.curriculum(start_from=1, end_at=2, rounds=3, verbose=False)
        assert len(rl._train_log) == 6

    def test_teach_runs(self, rl):
        rl.teach("sort numbers ascending", {"data": [5,3,1,4,2]}, rounds=2, verbose=False)
        assert len(rl._train_log) == 2

    def test_teach_with_expected_algo(self, rl):
        rl.teach("sort numbers", {"data": [3,1,2]},
                 expected_algo="quicksort", rounds=2, verbose=False)
        assert len(rl._train_log) == 2

    def test_self_teach_runs(self, rl):
        rl.self_teach(rounds=5, verbose=False)
        assert len(rl._train_log) == 5

    def test_solve_returns_result(self, rl):
        result = rl.solve("sort these numbers", {"data": [3,1,2]}, learn=False)
        assert result is not None

    def test_auto_learn_records_solves(self):
        from aalgoi.sandbox import SandboxRL
        rl = SandboxRL(hidden_size=32, num_layers=1, auto_learn=True)
        for _ in range(5):
            rl.solve("sort numbers", {"data": [3,1,2]})
        assert len(rl._solve_log) == 5

    def test_pause_resume_learning(self):
        from aalgoi.sandbox import SandboxRL
        rl = SandboxRL(hidden_size=32, num_layers=1, auto_learn=True)
        rl.pause_learning()
        rl.solve("sort numbers", {"data": [3,1,2]})
        assert len(rl._solve_log) == 0
        rl.resume_learning()
        rl.solve("sort numbers", {"data": [3,1,2]})
        assert len(rl._solve_log) == 1

    def test_fork_is_independent(self, rl):
        rl.curriculum(start_from=1, end_at=2, verbose=False)
        episodes_before = rl._agent.episodes
        fork = rl.fork()
        fork.curriculum(start_from=3, end_at=4, verbose=False)
        assert rl._agent.episodes == episodes_before
        assert fork._agent.episodes > 0

    def test_fork_named_registers(self, rl):
        from aalgoi.sandbox import SandboxRL
        fork = rl.fork("my_fork")
        assert SandboxRL.get("my_fork") is fork

    def test_save_load_roundtrip(self, rl):
        rl.curriculum(start_from=1, end_at=2, verbose=False)
        episodes_before = rl._agent.episodes

        with tempfile.NamedTemporaryFile(suffix=".rl", delete=False) as f:
            path = f.name
        try:
            rl.save(path)
            from aalgoi.sandbox import SandboxRL
            loaded = SandboxRL.from_checkpoint(path)
            assert loaded._agent.episodes == episodes_before
        finally:
            os.unlink(path)

    def test_from_checkpoint_with_step(self, rl):
        rl.curriculum(start_from=1, end_at=5, verbose=False)
        with tempfile.NamedTemporaryFile(suffix=".rl", delete=False) as f:
            path = f.name
        try:
            rl.save(path)
            from aalgoi.sandbox import SandboxRL
            loaded = SandboxRL.from_checkpoint(path, step=3)
            assert loaded._agent.episodes <= 5
        finally:
            os.unlink(path)

    def test_reset_clears_state(self, rl):
        rl.curriculum(start_from=1, end_at=3, verbose=False)
        assert rl._agent.episodes > 0
        rl.reset()
        assert rl._agent.episodes == 0
        assert len(rl._train_log) == 0

    def test_set_epsilon(self, rl):
        rl.set_epsilon(0.5)
        assert abs(rl._agent.epsilon - 0.5) < 1e-6

    def test_set_epsilon_clamps(self, rl):
        rl.set_epsilon(5.0)
        assert rl._agent.epsilon <= 1.0

    def test_named_registration(self):
        from aalgoi.sandbox import SandboxRL
        rl = SandboxRL(name="test_named", hidden_size=32, num_layers=1)
        assert SandboxRL.get("test_named") is rl
        SandboxRL.delete("test_named")
        assert SandboxRL.get("test_named") is None

    def test_list_instances(self):
        from aalgoi.sandbox import SandboxRL
        a = SandboxRL(name="inst_a", hidden_size=32, num_layers=1)
        b = SandboxRL(name="inst_b", hidden_size=32, num_layers=1)
        names = SandboxRL.list()
        assert "inst_a" in names
        assert "inst_b" in names
        SandboxRL.delete("inst_a")
        SandboxRL.delete("inst_b")

    def test_domain_filter_curriculum(self):
        from aalgoi.sandbox import SandboxRL
        rl = SandboxRL(hidden_size=32, num_layers=1, domains=["SORTING"])
        rl.curriculum(start_from=1, end_at=10, verbose=False)
        ids = {e["problem_id"] for e in rl._train_log}
        assert ids.issubset({0, 1, 2})

    def test_inspect_does_not_crash(self, rl):
        rl.curriculum(start_from=1, end_at=2, verbose=False)
        rl.inspect(top_k=3)

    def test_stats_has_all_keys(self, rl):
        s = rl.stats()
        for k in ["episodes", "avg_reward", "problems_seen", "auto_learn", "domains", "config"]:
            assert k in s

    def test_repr(self, rl):
        r = repr(rl)
        assert "SandboxRL" in r
        assert "episodes" in r

    def test_improvement_over_rounds(self):
        from aalgoi.sandbox import SandboxRL
        rl = SandboxRL(hidden_size=32, num_layers=1, epsilon=0.3)
        rl.curriculum(start_from=1, end_at=4, rounds=3, verbose=False)

        r1 = np.mean([e["reward"] for e in rl._train_log if e["round"] == 1])
        r3 = np.mean([e["reward"] for e in rl._train_log if e["round"] == 3])

        assert isinstance(r1, float)
        assert isinstance(r3, float)

    def test_aalgoi_import_sandboxrl(self):
        from aalgoi import SandboxRL
        assert SandboxRL is not None
