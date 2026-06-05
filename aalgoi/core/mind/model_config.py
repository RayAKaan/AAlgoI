from dataclasses import dataclass


@dataclass(frozen=True)
class MindConfig:
    vocab_size: int = 2048
    max_seq_len: int = 512
    hidden_dim: int = 384
    n_heads: int = 6
    n_layers: int = 4
    ffn_dim: int = 1536
    dropout: float = 0.1

    n_cognitive_actions: int = 25
    n_action_params: int = 64

    n_algorithms: int = 1000
    n_principles: int = 8
    algo_emb_dim: int = 384

    state_history_len: int = 20
    structural_feature_dim: int = 64

    lr: float = 3e-4
    clip_epsilon: float = 0.2
    entropy_coeff: float = 0.01
    value_coeff: float = 0.5
    max_grad_norm: float = 0.5
    ppo_epochs: int = 4
    batch_size: int = 32

    update_every_n_solves: int = 20
    checkpoint_every_n: int = 50

    use_fp16: bool = True

    def verify_size(self) -> dict:
        token_emb = self.vocab_size * self.hidden_dim
        pos_emb = self.max_seq_len * self.hidden_dim

        attn_per_layer = 4 * (self.hidden_dim * self.hidden_dim)
        ffn_per_layer = (
            self.hidden_dim * self.ffn_dim + self.ffn_dim * self.hidden_dim
        )
        ln_per_layer = 4 * self.hidden_dim
        total_per_layer = attn_per_layer + ffn_per_layer + ln_per_layer

        transformer_total = self.n_layers * total_per_layer

        algo_emb = self.n_algorithms * self.algo_emb_dim
        princ_emb = self.n_principles * self.hidden_dim

        struct_enc = self.structural_feature_dim * self.hidden_dim

        gru_params = (
            3 * self.hidden_dim * self.n_cognitive_actions
            + 3 * self.hidden_dim * self.hidden_dim
            + 3 * self.hidden_dim
        )

        policy_head = (
            self.hidden_dim * self.hidden_dim
            + self.hidden_dim * self.n_cognitive_actions
        )

        value_head = self.hidden_dim * (self.hidden_dim // 2) + (self.hidden_dim // 2)

        param_heads = (
            self.hidden_dim * self.n_algorithms
            + self.hidden_dim * 16
            + self.hidden_dim * 8
            + self.hidden_dim * self.n_principles
        )

        total_params = (
            token_emb + pos_emb
            + transformer_total
            + algo_emb + princ_emb
            + struct_enc + gru_params
            + policy_head + value_head
            + param_heads
        )

        bytes_fp32 = total_params * 4
        bytes_fp16 = total_params * 2

        result = {
            "total_params": total_params,
            "size_fp32_mb": bytes_fp32 / (1024**2),
            "size_fp16_mb": bytes_fp16 / (1024**2),
            "within_target": 10 <= bytes_fp16 / (1024**2) <= 20,
            "breakdown": {
                "embeddings_mb": (token_emb + pos_emb + algo_emb + princ_emb) * 2 / (1024**2),
                "transformer_mb": transformer_total * 2 / (1024**2),
                "heads_mb": (policy_head + value_head + param_heads) * 2 / (1024**2),
                "other_mb": (struct_enc + gru_params) * 2 / (1024**2),
            },
        }

        assert result["within_target"], (
            f"Model size {result['size_fp16_mb']:.1f}MB outside 10-20MB target."
        )

        return result


DEFAULT_CONFIG = MindConfig()
