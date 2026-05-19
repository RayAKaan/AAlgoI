"""
Self-Play Training (4 Hours)

3-phase adversarial training:
  Warmup (2k) -> Exploration (5k) -> Exploitation (3k)

Usage:
    python training/self_play_train.py
"""

import torch
import logging
from core.rl.agents.selection_agent import PPOAgent
from core.rl.powerhouse_agent import WorldModel
from training.self_play import SelfPlayEngine

logger = logging.getLogger(__name__)


def self_play_train():
    logger.info("=" * 60)
    logger.info("AAlgoI Self-Play Training (4 Hours)")
    logger.info("=" * 60)

    agent = PPOAgent(state_dim=200, num_actions=20)
    world_model = WorldModel(state_dim=200, action_dim=20)
    self_play = SelfPlayEngine(agent, world_model)

    phases = [
        {"name": "Warmup", "iterations": 2000},
        {"name": "Exploration", "iterations": 5000},
        {"name": "Exploitation", "iterations": 3000},
    ]

    for phase in phases:
        logger.info("--- Phase: %s ---", phase["name"])
        stats = self_play.train_round(
            iterations=phase["iterations"],
            state_dim=200,
        )
        logger.info("Results: Wins=%s, Losses=%s", stats.get("wins", 0), stats.get("losses", 0))
        agent.save(f"checkpoints/self_play_{phase['name'].lower()}.pt")

    agent.save("checkpoints/self_play_final.pt")
    torch.save(world_model.state_dict(), "checkpoints/world_model.pt")

    logger.info("Self-play training complete!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    self_play_train()
