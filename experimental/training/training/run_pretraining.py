#!/usr/bin/env python3
"""Pre-train the RL algorithm selector on synthetic data across all 3 domains."""

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

from training.pretrain import RLPretrainer


def main():
    config = {
        "episodes_per_level": 800,
        "batch_size": 128,
        "learning_rate": 3e-4,
        "save_dir": "checkpoints/pretrain_v2",
        "time_budget_ms": 500.0,
    }

    trainer = RLPretrainer(save_dir=config["save_dir"], config=config)
    summary = trainer.run_training(total_episodes=2000)

    print("\n=== Training Summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
