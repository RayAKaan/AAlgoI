"""
Federated Knowledge Sync
Enables multiple AAlgoI instances to share learnings without sharing data.
Two modes:
  - central: Client-server mode via aalgoi.org API
  - p2p: Peer-to-peer mode (original implementation)
"""

import hashlib
import logging
import threading
import time
import uuid
from typing import Any

logger = logging.getLogger(__name__)


class FederatedKnowledgeSync:
    """
    Federated learning for algorithms.
    Users share "what worked", not "what the data was".
    """

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self.enabled = self.config.get("enabled", False)
        self.mode = self.config.get("mode", "central")
        self.server_url = self.config.get(
            "server_url", "https://api.aalgoi.org/federated"
        )
        self.sync_interval = self.config.get("sync_interval", 3600)
        self.anonymous_id = self._get_anonymous_id()

        # P2P fields (legacy)
        self.node_urls: list[str] = self.config.get("nodes", [])
        self.local_kb: Any = None
        self._pending_updates: list[dict] = []
        self._last_sync: float = 0.0
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None

    def _get_anonymous_id(self) -> str:
        """Generate anonymous client ID (MAC hash + UUID)."""
        try:
            mac = ":".join(
                [
                    f"{(uuid.getnode() >> elements) & 0xFF:02x}"
                    for elements in range(0, 2 * 6, 2)
                ][::-1]
            )
            return hashlib.sha256(f"{mac}-{uuid.uuid4()}".encode()).hexdigest()[:16]
        except Exception:
            return str(uuid.uuid4())[:16]

    # ============================================
    # CENTRAL-SERVER MODE
    # ============================================

    def push_learnings(self, knowledge_update: dict) -> bool:
        """
        Push local model updates to the global network.
        Only sends metrics/metadata, NEVER raw user data.
        """
        if not self.enabled or self.mode != "central":
            return False

        payload = {
            "client_id": self.anonymous_id,
            "timestamp": time.time(),
            "updates": knowledge_update,
        }

        try:
            import requests

            response = requests.post(
                f"{self.server_url}/contribute", json=payload, timeout=5
            )
            if response.status_code == 200:
                logger.info("Successfully pushed learnings to global network")
                return True
            else:
                logger.debug("Push failed: HTTP %s", response.status_code)
        except Exception as e:
            logger.debug("Push failed (offline?): %s", e)

        return False

    def pull_global_knowledge(self) -> dict | None:
        """
        Download aggregated knowledge from the network.
        Returns dict of {algorithm_name: {avg_reward, success_rate, ...}}
        """
        if not self.enabled or self.mode != "central":
            return None

        try:
            import requests

            response = requests.get(
                f"{self.server_url}/aggregate",
                params={"client_id": self.anonymous_id},
                timeout=10,
            )
            if response.status_code == 200:
                data = response.json()
                logger.info("Pulled %d global knowledge entries", len(data))
                return data
            else:
                logger.debug("Pull failed: HTTP %s", response.status_code)
        except Exception as e:
            logger.debug("Pull failed (offline?): %s", e)

        return None

    def sync(self, local_kb) -> bool:
        """
        Full sync: Push local best, pull global best, merge.
        """
        if not self.enabled:
            return False

        local_best = local_kb.get_top_performing(n=10) if hasattr(local_kb, "get_top_performing") else {}
        self.push_learnings(local_best)

        global_knowledge = self.pull_global_knowledge()

        if global_knowledge and hasattr(local_kb, "merge"):
            local_kb.merge(global_knowledge)
            self._last_sync = time.time()
            return True

        return False

    # ============================================
    # P2P MODE (legacy)
    # ============================================

    def broadcast_update(self, algo_name: str, context_signature: str, score: float):
        if self.mode != "p2p":
            return
        payload = {
            "algo": algo_name,
            "context_sig": context_signature,
            "score": score,
            "timestamp": time.time(),
            "node_id": id(self),
        }
        with self._lock:
            self._pending_updates.append(payload)

    def start_background_sync(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._sync_loop, daemon=True)
        self._thread.start()

    def stop_background_sync(self):
        self._running = False

    def _sync_loop(self):
        while self._running:
            self.sync_now()
            time.sleep(self.sync_interval)

    def sync_now(self):
        if not self.node_urls:
            return

        with self._lock:
            updates = list(self._pending_updates)
            self._pending_updates.clear()

        for url in self.node_urls:
            try:
                import requests

                if updates:
                    requests.post(
                        f"{url}/knowledge/update",
                        json={"updates": updates},
                        timeout=2,
                    )
                response = requests.get(f"{url}/knowledge/recent", timeout=2)
                if response.status_code == 200:
                    data = response.json()
                    remote_updates = data.get("updates", [])
                    if self.local_kb and remote_updates:
                        self._incorporate_remote(remote_updates)
            except Exception:
                pass

        self._last_sync = time.time()

    def _incorporate_remote(self, updates: list[dict]):
        pass

    def set_local_kb(self, kb: Any):
        self.local_kb = kb

    def add_node(self, url: str):
        if url not in self.node_urls:
            self.node_urls.append(url)

    def remove_node(self, url: str):
        if url in self.node_urls:
            self.node_urls.remove(url)

    def get_stats(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "enabled": self.enabled,
            "nodes": self.node_urls,
            "sync_interval": self.sync_interval,
            "last_sync": self._last_sync,
            "pending_updates": len(self._pending_updates),
            "running": self._running,
        }
