"""
Federated Training (Continuous)

Runs in background, syncing knowledge with the global network
every hour. Learns from all users without sharing raw data.

Usage:
    trainer = FederatedTrainer(solver)
    trainer.start()
    # ... user continues working ...
    trainer.stop()
"""

import logging
import threading
import time

logger = logging.getLogger(__name__)


class FederatedTrainer:
    """
    Background federated trainer.
    Pulls global knowledge → merges locally → pushes local best.
    Runs on a daemon thread at configurable intervals (default 1 hour).
    """

    def __init__(self, solver):
        self.solver = solver
        self.running = False
        self.thread: threading.Thread = None
        self.sync_interval = 3600

    def start(self):
        """Start background federated sync."""
        self.running = True
        self.thread = threading.Thread(target=self._training_loop, daemon=True)
        self.thread.start()
        logger.info("Federated training started (interval=%ds)", self.sync_interval)

    def stop(self):
        """Stop background federated sync."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        logger.info("Federated training stopped")

    def _training_loop(self):
        while self.running:
            try:
                global_knowledge = self.solver.federated.pull_global_knowledge()
                if global_knowledge:
                    self.solver.knowledge_base.merge(global_knowledge)
                    logger.info("Merged %d federated entries", len(global_knowledge))

                local_best = self.solver.knowledge_base.get_top_performing(n=10)
                self.solver.federated.push_learnings(local_best)

            except Exception as e:
                logger.debug("Federated sync error: %s", e)

            for _ in range(self.sync_interval):
                if not self.running:
                    break
                time.sleep(1)
