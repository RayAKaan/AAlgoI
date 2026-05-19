import hashlib
import json
import os
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


@dataclass
class AlgorithmMetadata:
    """Metadata for RL-discovered algorithms (no data, just description)."""
    name: str
    use_case: str
    problem_type: str
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    complexity: str = "Unknown"
    discovery_date: str = field(default_factory=lambda: datetime.now().isoformat())
    discovered_by: str = "RL-Agent"
    training_episodes: int = 0
    avg_reward: float = 0.0
    tags: List[str] = field(default_factory=list)
    is_verified: bool = False


class AlgorithmMarketplace:
    """
    Global registry for RL-discovered and community-contributed algorithms.
    Stores algorithm code + metadata, but NEVER training data.
    """

    def __init__(self, storage_path: str = "~/.aalgoi/marketplace"):
        self.storage_path = os.path.expanduser(storage_path)
        os.makedirs(self.storage_path, exist_ok=True)

        self.algorithms: Dict[str, AlgorithmMetadata] = {}
        self.code_cache: Dict[str, str] = {}

        self._load_from_disk()

    def _load_from_disk(self):
        index_path = os.path.join(self.storage_path, "index.json")
        if os.path.exists(index_path):
            try:
                with open(index_path, "r") as f:
                    data = json.load(f)
                    for name, meta_dict in data.items():
                        self.algorithms[name] = AlgorithmMetadata(**meta_dict)
            except Exception as e:
                logger.warning("Failed to load marketplace index: %s", e)

        for filename in os.listdir(self.storage_path):
            if filename.endswith(".py") and filename != "index.json":
                name = filename[:-3]
                try:
                    with open(os.path.join(self.storage_path, filename), "r") as f:
                        self.code_cache[name] = f.read()
                except Exception as e:
                    logger.warning("Failed to load algorithm %s: %s", filename, e)

    def _save_to_disk(self):
        index_path = os.path.join(self.storage_path, "index.json")
        with open(index_path, "w") as f:
            json.dump(
                {name: asdict(meta) for name, meta in self.algorithms.items()},
                f, indent=2,
            )

        for name, code in self.code_cache.items():
            code_path = os.path.join(self.storage_path, f"{name}.py")
            with open(code_path, "w") as f:
                f.write(code)

    def register_algorithm(
        self, name: str, code: str, metadata: AlgorithmMetadata
    ) -> str:
        """
        Register a new algorithm discovered by RL or contributed by user.
        Stores code + metadata, never data.
        Returns the full registered name with ID.
        """
        algo_id = hashlib.sha256(code.encode()).hexdigest()[:16]
        full_name = f"{name}_{algo_id}"

        self.algorithms[full_name] = metadata
        self.code_cache[full_name] = code
        self._save_to_disk()

        logger.info(
            "Registered algorithm: %s (use case: %s)",
            full_name, metadata.use_case,
        )
        return full_name

    def find_by_use_case(self, use_case: str) -> List[AlgorithmMetadata]:
        """Find algorithms by use case description (keyword match)."""
        matches = []
        use_case_lower = use_case.lower()
        for meta in self.algorithms.values():
            if use_case_lower in meta.use_case.lower():
                matches.append(meta)
            elif any(tag in use_case_lower for tag in meta.tags):
                matches.append(meta)
        return sorted(matches, key=lambda m: m.avg_reward, reverse=True)

    def load_algorithm(self, name: str):
        """Dynamically load algorithm code into runtime."""
        if name not in self.code_cache:
            return None

        code = self.code_cache[name]
        namespace = {}
        try:
            exec(code, namespace)
            for obj in namespace.values():
                if isinstance(obj, type) and hasattr(obj, "process"):
                    return obj()
        except Exception as e:
            logger.error("Failed to load algorithm %s: %s", name, e)
            return None
        return None

    def list_algorithms(self) -> List[Dict[str, Any]]:
        """List all registered algorithms with their metadata."""
        return [
            {
                "name": name,
                "use_case": meta.use_case,
                "problem_type": meta.problem_type,
                "avg_reward": meta.avg_reward,
                "discovered_by": meta.discovered_by,
                "is_verified": meta.is_verified,
            }
            for name, meta in self.algorithms.items()
        ]
