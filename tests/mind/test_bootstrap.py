import pytest
import torch
from pathlib import Path

from aalgoi.core.mind.training.bootstrap_trainer import (
    BootstrapTrainer,
    get_ideal_paths,
    generate_training_data,
    TrainingStep,
    IdealPath,
    _generate_problem_text,
    _generate_problem_data,
)
from aalgoi.core.mind.rl_mind import AlgorithmicMind
from aalgoi.core.mind.knowledge_graph import AlgorithmicKnowledgeGraph
from aalgoi.core.mind.model_config import MindConfig


@pytest.fixture
def kg(tmp_path):
    return AlgorithmicKnowledgeGraph(tmp_path / "kg_bootstrap")


@pytest.fixture
def config():
    return MindConfig()


@pytest.fixture
def mind(config):
    return AlgorithmicMind(config)


@pytest.fixture
def trainer(mind, kg, config):
    return BootstrapTrainer(mind, kg, config)


class TestIdealPaths:
    def test_paths_exist(self):
        paths = get_ideal_paths()
        assert len(paths) > 0

    def test_all_paths_have_actions(self):
        paths = get_ideal_paths()
        for path in paths:
            assert len(path.ideal_actions) > 0

    def test_all_paths_have_algorithm(self):
        paths = get_ideal_paths()
        for path in paths:
            assert path.best_algorithm != ""

    def test_paths_cover_domains(self):
        paths = get_ideal_paths()
        domains = set(p.domain for p in paths)
        assert "graph" in domains
        assert "text" in domains

    def test_paths_have_reasonable_iterations(self):
        paths = get_ideal_paths()
        for path in paths:
            assert 3 <= path.expected_iterations <= 15


class TestTrainingDataGeneration:
    def test_generate_produces_trajectories(self, kg, config):
        trajectories = generate_training_data(kg, config, n_augmentations=2)
        assert len(trajectories) > 0

    def test_trajectories_have_steps(self, kg, config):
        trajectories = generate_training_data(kg, config, n_augmentations=1)
        for traj in trajectories:
            assert len(traj.steps) > 0

    def test_steps_have_features(self, kg, config):
        trajectories = generate_training_data(kg, config, n_augmentations=1)
        for traj in trajectories:
            for step in traj.steps:
                assert step.data_features is not None
                assert step.data_features.shape[0] == 200

    def test_steps_have_rewards(self, kg, config):
        trajectories = generate_training_data(kg, config, n_augmentations=1)
        for traj in trajectories:
            for step in traj.steps:
                assert step.reward >= 0

    def test_augmentation_creates_variety(self, kg, config):
        trajectories = generate_training_data(kg, config, n_augmentations=3)
        texts = set()
        for traj in trajectories:
            if traj.steps:
                texts.add(traj.steps[0].problem_text)
        assert len(texts) > 1

    def test_total_reward_positive(self, kg, config):
        trajectories = generate_training_data(kg, config, n_augmentations=1)
        for traj in trajectories:
            assert traj.total_reward > 0


class TestProblemGeneration:
    def test_text_generation(self):
        paths = get_ideal_paths()
        for path in paths[:5]:
            text = _generate_problem_text(path, 0)
            assert text is not None
            assert len(text) > 0

    def test_text_varies_with_augmentation(self):
        paths = get_ideal_paths()
        path = paths[0]
        texts = set()
        for aug in range(5):
            texts.add(_generate_problem_text(path, aug))
        assert len(texts) > 1

    def test_data_generation(self):
        paths = get_ideal_paths()
        for path in paths[:5]:
            data = _generate_problem_data(path, 0)
            assert data is not None

    def test_data_varies_with_augmentation(self):
        paths = get_ideal_paths()
        path = paths[0]
        data0 = _generate_problem_data(path, 0)
        data1 = _generate_problem_data(path, 1)
        assert data0 != data1


class TestBootstrapTraining:
    def test_train_runs(self, trainer):
        stats = trainer.train(n_epochs=2, n_augmentations=1, batch_size=4)
        assert "n_trajectories" in stats
        assert "n_steps" in stats
        assert stats["n_trajectories"] > 0
        assert stats["n_steps"] > 0

    def test_train_records_phase2(self, trainer):
        stats = trainer.train(n_epochs=4, n_augmentations=1, batch_size=4)
        assert "phase2_reward" in stats
        assert len(stats["phase2_reward"]) > 0

    def test_train_updates_model_weights(self, mind, kg, config):
        trainer = BootstrapTrainer(mind, kg, config)
        weights_before = {k: v.clone() for k, v in mind.state_dict().items()}

        trainer.train(n_epochs=3, n_augmentations=2, batch_size=8)

        changed = 0
        for k, v in mind.state_dict().items():
            if k in weights_before and not torch.equal(v, weights_before[k]):
                changed += 1
        assert changed > 0

    def test_evaluate_runs(self, trainer):
        results = trainer.evaluate(n_samples=10)
        assert "accuracy" in results
        assert "correct" in results
        assert "total" in results
        assert results["total"] > 0

    def test_evaluate_accuracy_between_0_and_1(self, trainer):
        trainer.train(n_epochs=3, n_augmentations=2, batch_size=8)
        results = trainer.evaluate(n_samples=20)
        assert 0.0 <= results["accuracy"] <= 1.0


class TestBehavioralCloningLoss:
    def test_loss_computation(self, trainer, kg, config):
        trajectories = generate_training_data(kg, config, n_augmentations=1)
        if not trajectories:
            pytest.skip("No trajectories generated")

        steps = trajectories[0].steps[:4]
        loss = trainer._behavioral_cloning_loss(steps)
        assert loss is not None
        assert loss.item() >= 0

    def test_loss_is_differentiable(self, trainer, kg, config):
        trajectories = generate_training_data(kg, config, n_augmentations=1)
        if not trajectories:
            pytest.skip("No trajectories generated")

        steps = trajectories[0].steps[:4]
        loss = trainer._behavioral_cloning_loss(steps)
        loss.backward()

        has_grad = False
        for p in trainer.mind.parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                has_grad = True
                break
        assert has_grad
