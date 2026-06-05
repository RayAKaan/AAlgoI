import hashlib
import json
import logging
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests

from aalgoi.core.token_manager import TokenManager

logger = logging.getLogger(__name__)

# ── Defaults (overridden by config.json) ─────────────────────────────────────

REGISTRY_PROD_BASE = "https://aalgoi-federation.fly.dev"
REGISTRY_DEV_BASE = (
    "https://raw.githubusercontent.com"
    "/aalgoi/algorithm-registry/main"
)
_DEFAULT_REGISTRY_BASE = REGISTRY_PROD_BASE
_DEFAULT_API_BASE = REGISTRY_PROD_BASE
_DEFAULT_SYNC_INTERVAL = 6 * 3600
_DEFAULT_BACKOFF_BASE = 60
_DEFAULT_BACKOFF_MAX = 3600

# ── Helpers ──────────────────────────────────────────────────────────────────


def _load_registry_config() -> dict:
    """Load registry config from config.json, returning only the registry block."""
    try:
        # Search relative to this file's project root, then cwd, then env var
        candidates = [
            Path(__file__).resolve().parent.parent / "config" / "config.json",
            Path.cwd() / "config" / "config.json",
        ]
        env_path = os.getenv("AALGOI_CONFIG")
        if env_path:
            candidates.insert(0, Path(env_path))

        for path in candidates:
            if path.exists():
                raw = json.loads(path.read_text())
                return raw.get("registry", {})
    except Exception as exc:
        logger.debug("Could not load registry config: %s", exc)
    return {}


# ── Sync class ───────────────────────────────────────────────────────────────


class GitHubRegistrySync:

    def __init__(
        self,
        local_registry,
        embedder,
        agent=None,
        cache_dir: str = "~/.aalgoi/registry",
        config: Optional[dict] = None,
    ):
        self.registry = local_registry
        self.embedder = embedder
        self.agent = agent
        self.cache_dir = Path(cache_dir).expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.cache_dir / "sync_state.json"

        # Load config
        cfg = _load_registry_config() if config is None else config
        self.registry_base = cfg.get("base_url", _DEFAULT_REGISTRY_BASE)
        self.api_base = cfg.get("api_url", _DEFAULT_API_BASE)
        self.sync_interval = cfg.get("sync_interval_seconds", _DEFAULT_SYNC_INTERVAL)
        self.backoff_base = cfg.get("backoff_base_seconds", _DEFAULT_BACKOFF_BASE)
        self.backoff_max = cfg.get("backoff_max_seconds", _DEFAULT_BACKOFF_MAX)

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._backoff_delay = 0
        self._push_queue: list[dict] = []

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._sync_loop,
            daemon=True,
            name="aalgoi-registry-sync",
        )
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    # ── Public API ─────────────────────────────────────────────────────────

    def sync_now(self) -> dict:
        return self._pull()

    def sync_pull(self) -> dict:
        return self._pull()

    def status(self) -> dict:
        state = self._load_state()
        return {
            "last_sync": state.get("last_sync", "never"),
            "index_version": state.get("index_version", 0),
            "algorithm_count": len(state.get("algorithm_names", [])),
            "backoff_seconds": self._backoff_delay,
            "push_queue_depth": len(self._push_queue),
            "registry_base": self.registry_base,
            "api_base": self.api_base,
            "sync_interval": self.sync_interval,
        }

    # ── Pull ───────────────────────────────────────────────────────────────

    def _pull(self) -> dict:
        try:
            remote_index = self._fetch_index()
        except Exception as e:
            self._backoff_delay = self._backoff_delay or self.backoff_base
            backoff = min(self._backoff_delay, self.backoff_max)
            logger.warning(
                "Sync paused: GitHub unreachable (%s). "
                "Will retry in %ds. "
                "Run `aalgoi sync status` to check connectivity.",
                e,
                backoff,
            )
            self._backoff_delay = min(backoff * 2, self.backoff_max)
            return {"status": "offline", "error": str(e)}

        # Success — reset backoff
        self._backoff_delay = 0

        local_state = self._load_state()
        local_version = local_state.get("index_version", 0)
        remote_version = remote_index.get("version", 0)

        if remote_version <= local_version:
            self._flush_queue()
            return {"status": "up_to_date", "version": local_version}

        local_names = set(local_state.get("algorithm_names", []))
        new_algos = [
            a for a in remote_index.get("algorithms", [])
            if a["name"] not in local_names
        ]

        results = [self._download_and_register(a) for a in new_algos]
        success = [r for r in results if r["status"] == "registered"]

        self._save_state({
            "index_version": remote_version,
            "last_sync": datetime.now().isoformat(),
            "algorithm_names": [
                a["name"] for a in remote_index.get("algorithms", [])
            ],
        })

        logger.info(
            "Registry sync complete: %d new, %d failed, %d total",
            len(success),
            len(results) - len(success),
            remote_version,
        )

        self._flush_queue()

        return {
            "status": "updated",
            "new_algorithms": len(success),
            "failed": len(results) - len(success),
            "details": results,
        }

    def _fetch_index(self) -> dict:
        return self._fetch_json("index.json")

    def _download_and_register(self, algo_meta: dict) -> dict:
        name = algo_meta["name"]
        path = algo_meta["path"]

        try:
            code_url = f"{self.registry_base}/{path}.py"
            code_resp = requests.get(code_url, timeout=10)
            code_resp.raise_for_status()
            code = code_resp.text

            computed = "sha256:" + hashlib.sha256(code.encode()).hexdigest()
            if computed != algo_meta.get("checksum", computed):
                return {"name": name, "status": "failed", "reason": "checksum_mismatch"}

            meta_url = f"{self.registry_base}/{path}.json"
            meta_resp = requests.get(meta_url, timeout=10)
            meta_resp.raise_for_status()
            metadata = meta_resp.json()

            cache_path = self.cache_dir / f"{name}.py"
            cache_path.write_text(code)

            self.registry.register_from_code(name, code)

            if "embedding" in metadata and self.embedder:
                import torch
                embed = torch.tensor(metadata["embedding"], dtype=torch.float32)
                self.embedder.add_embedding(name, embed)

            if self.agent:
                all_embeds = self.embedder.get_all_embeddings(self.registry)
                self.agent.update_algo_embeddings(
                    all_embeds,
                    list(self.registry.keys()),
                )

            return {"name": name, "status": "registered"}

        except Exception as e:
            return {"name": name, "status": "failed", "reason": str(e)}

    # ── Push ───────────────────────────────────────────────────────────────

    def push_discovered(
        self,
        algorithm,
        validation_result,
        github_token: str = "",
    ) -> dict:
        if not validation_result.passed:
            return {"status": "rejected", "reason": "failed_validation"}

        # Resolve token: explicit arg > env var > queue
        token = github_token or TokenManager.get_token()
        if not token:
            self._push_queue.append({
                "algorithm": algorithm,
                "validation_result": validation_result,
            })
            logger.warning(
                "No GITHUB_TOKEN set. Queued push for '%s'. "
                "Set the env var and run `aalgoi sync status` to flush.",
                algorithm.name,
            )
            return {"status": "queued", "reason": "no_token"}

        if self._backoff_delay > 0:
            self._push_queue.append({
                "algorithm": algorithm,
                "validation_result": validation_result,
            })
            logger.warning(
                "Sync is paused (backoff %ds). Queued push for '%s'. "
                "Will retry automatically.",
                self._backoff_delay,
                algorithm.name,
            )
            return {"status": "queued", "reason": "offline"}

        return self._do_push(algorithm, validation_result, token)

    def _do_push(self, algorithm, validation_result, token: str) -> dict:
        branch = f"discovered/{algorithm.name.lower()}"
        headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json",
        }

        try:
            ref = requests.get(
                f"{self.api_base}/git/ref/heads/main",
                headers=headers,
            ).json()
            sha = ref["object"]["sha"]

            requests.post(
                f"{self.api_base}/git/refs",
                headers=headers,
                json={"ref": f"refs/heads/{branch}", "sha": sha},
            )

            import base64
            code_b64 = base64.b64encode(
                algorithm.serialize().encode()
            ).decode()

            requests.put(
                f"{self.api_base}/contents/algorithms/discovered/{algorithm.name}.py",
                headers=headers,
                json={
                    "message": f"Add: {algorithm.name}",
                    "content": code_b64,
                    "branch": branch,
                },
            )

            metadata = self._build_metadata(algorithm, validation_result)
            meta_b64 = base64.b64encode(
                json.dumps(metadata, indent=2).encode()
            ).decode()

            requests.put(
                f"{self.api_base}/contents/algorithms/discovered/{algorithm.name}.json",
                headers=headers,
                json={
                    "message": f"Metadata: {algorithm.name}",
                    "content": meta_b64,
                    "branch": branch,
                },
            )

            pr = requests.post(
                f"{self.api_base}/pulls",
                headers=headers,
                json={
                    "title": (
                        f"[Auto] {algorithm.name} "
                        f"+{validation_result.improvement_pct:.1f}% "
                        f"on {algorithm.problem_type}"
                    ),
                    "body": self._build_pr_body(algorithm, validation_result),
                    "head": branch,
                    "base": "main",
                },
            ).json()

            return {
                "status": "submitted",
                "pr_url": pr.get("html_url", ""),
            }

        except Exception as e:
            logger.error("Push failed for %s: %s", algorithm.name, e)
            return {"status": "failed", "error": str(e)}

    def _flush_queue(self):
        if not self._push_queue:
            return
        token = TokenManager.get_token()
        if not token:
            return

        remaining = []
        for entry in self._push_queue:
            algo = entry["algorithm"]
            vr = entry["validation_result"]
            result = self._do_push(algo, vr, token)
            if result.get("status") == "submitted":
                logger.info("Flushed queued push: %s → %s", algo.name, result.get("pr_url", "?"))
            else:
                remaining.append(entry)

        self._push_queue[:] = remaining
        if remaining:
            logger.info("%d pushes still queued after flush", len(remaining))

    # ── Sync loop with backoff ────────────────────────────────────────────

    def _sync_loop(self):
        time.sleep(10)
        self._pull()

        while self._running:
            delay = self.sync_interval
            if self._backoff_delay > 0:
                delay = min(self._backoff_delay, self.backoff_max)
            time.sleep(delay)
            if self._running:
                self._pull()

    # ── Internal helpers ──────────────────────────────────────────────────

    def _fetch_json(self, path: str) -> dict:
        url = f"{self.registry_base}/{path}"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        return resp.json()

    def _load_state(self) -> dict:
        if self.state_path.exists():
            return json.loads(self.state_path.read_text())
        return {}

    def _save_state(self, state: dict):
        self.state_path.write_text(json.dumps(state, indent=2))

    def _build_metadata(self, algorithm, validation_result) -> dict:
        return {
            "name": algorithm.name,
            "version": "1.0.0",
            "status": "beta",
            "problem_types": [algorithm.problem_type],
            "embedding": algorithm.embedding.tolist(),
            "benchmark": {
                "improvement_pct": validation_result.improvement_pct,
                "baseline": validation_result.baseline_name,
                "pass_rate": validation_result.pass_rate,
            },
            "discovered_by": "rl_agent",
            "submitted_at": datetime.now().isoformat(),
        }

    def _build_pr_body(self, algorithm, validation_result) -> str:
        return f"""
## Auto-Discovered: `{algorithm.name}`

**Problem Type:** {algorithm.problem_type}
**Improvement over baseline:** +{validation_result.improvement_pct:.1f}%
**Validation pass rate:** {validation_result.pass_rate:.2%}

| Test | Result |
|------|--------|
| Correctness | {validation_result.results['correctness']:.2%} |
| Edge cases | {'✅' if validation_result.results['edge_cases']['all_passed'] else '❌'} |
| Beats baseline | {'✅' if validation_result.results['performance']['beats_baseline'] else '❌'} |

*Opened automatically by the AAlgoI algorithm discovery engine.*
        """.strip()
