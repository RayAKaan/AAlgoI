from aalgoi.core.mind.model_config import MindConfig


class TestMindConfig:
    def test_default_config_creates(self):
        cfg = MindConfig()
        assert cfg.hidden_dim == 384
        assert cfg.n_layers == 4
        assert cfg.n_heads == 6

    def test_size_within_target(self):
        cfg = MindConfig()
        info = cfg.verify_size()
        assert info["within_target"], (
            f"Size {info['size_fp16_mb']:.1f}MB outside 10-20MB target"
        )

    def test_size_breakdown_has_all_keys(self):
        cfg = MindConfig()
        info = cfg.verify_size()
        assert "breakdown" in info
        for key in ("embeddings_mb", "transformer_mb", "heads_mb", "other_mb"):
            assert key in info["breakdown"], f"Missing breakdown key: {key}"

    def test_total_params_positive(self):
        cfg = MindConfig()
        info = cfg.verify_size()
        assert info["total_params"] > 0
        assert 5_000_000 < info["total_params"] < 15_000_000
