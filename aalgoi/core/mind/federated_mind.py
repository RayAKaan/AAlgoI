import hashlib
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class StructuralHasher:
    def hash_algorithm(self, algorithm: dict) -> str:
        serialized = json.dumps(algorithm, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()

    def compute_signature(self, algorithm: dict) -> str:
        return self.hash_algorithm(algorithm)


class FederatedMindSync:
    def __init__(self, sync_dir: str = "~/.aalgoi/federation") -> None:
        self.sync_dir = Path(sync_dir).expanduser().resolve()
        self.outbox_dir = self.sync_dir / "outbox"
        self.inbox_dir = self.sync_dir / "inbox"
        self.outbox_dir.mkdir(parents=True, exist_ok=True)
        self.inbox_dir.mkdir(parents=True, exist_ok=True)
        self.hasher = StructuralHasher()
        self._max_messages = 10

    def anonymized_share(self, algorithm: dict) -> None:
        structural_hash = self.hasher.hash_algorithm(algorithm)
        signature = self.hasher.compute_signature(algorithm)
        payload = {
            "structural_hash": structural_hash,
            "signature": signature,
            "epsilon": 0.1,
        }
        timestamp = hashlib.md5(
            json.dumps(payload, sort_keys=True).encode()
        ).hexdigest()[:12]
        out_path = self.outbox_dir / f"share_{timestamp}.json"
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)

    def sync(self, max_messages: int | None = None) -> list[dict]:
        if max_messages is None:
            max_messages = self._max_messages

        inbox_files = sorted(self.inbox_dir.glob("*.json"))
        collected = []
        for fpath in inbox_files[:max_messages]:
            try:
                with open(fpath) as f:
                    data = json.load(f)
                collected.append(data)
                fpath.unlink(missing_ok=True)
            except (OSError, json.JSONDecodeError) as e:
                logger.warning("Failed to read inbox message %s: %s", fpath.name, e)
                fpath.unlink(missing_ok=True)

        return collected
