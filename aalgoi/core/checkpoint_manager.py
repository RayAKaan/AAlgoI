import json
from datetime import datetime
from pathlib import Path


class CheckpointManager:

    def __init__(self, base_dir: str = "~/.aalgoi"):
        self.base_dir = Path(base_dir).expanduser()
        self.adapter_dir = self.base_dir / "checkpoints" / "adapters"
        self.manifest_path = self.base_dir / "checkpoint_manifest.json"
        self.adapter_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(
        self,
        adapter,
        solve_count: int,
        metrics: dict,
    ) -> int:
        version = self._next_version()
        path = self.adapter_dir / f"adapter_v{version}.pt"

        adapter.save(str(path))

        manifest = self._load_manifest()
        manifest['checkpoints'].append({
            'version': version,
            'path': str(path),
            'solve_count': solve_count,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
        })
        manifest['current_version'] = version
        self._save_manifest(manifest)

        return version

    def rollback(self, version: int | None = None) -> str | None:
        manifest = self._load_manifest()

        if not manifest['checkpoints']:
            print("[Checkpoint] No checkpoints to roll back to.")
            return None

        if version is None:
            current = manifest['current_version']
            version = current - 1
            if version < 1:
                print("[Checkpoint] Already at oldest checkpoint.")
                return None

        entry = next(
            (c for c in manifest['checkpoints'] if c['version'] == version),
            None,
        )

        if not entry:
            print(f"[Checkpoint] Version {version} not found.")
            return None

        manifest['current_version'] = version
        self._save_manifest(manifest)

        print(f"[Checkpoint] Rolled back to v{version} "
              f"(solve_count={entry['solve_count']}, "
              f"timestamp={entry['timestamp'][:10]})")
        return entry['path']

    def reset(self) -> bool:
        import shutil
        if self.adapter_dir.exists():
            shutil.rmtree(self.adapter_dir)
        self.adapter_dir.mkdir(parents=True, exist_ok=True)

        self._save_manifest({'checkpoints': [], 'current_version': 0})
        print("[Checkpoint] Reset to base model.")
        return True

    def list_checkpoints(self) -> list:
        return self._load_manifest()['checkpoints']

    def get_current_adapter_path(self) -> str | None:
        manifest = self._load_manifest()
        version = manifest.get('current_version', 0)
        if version == 0:
            return None
        entry = next(
            (c for c in manifest['checkpoints'] if c['version'] == version),
            None,
        )
        return entry['path'] if entry else None

    def _next_version(self) -> int:
        manifest = self._load_manifest()
        if not manifest['checkpoints']:
            return 1
        return max(c['version'] for c in manifest['checkpoints']) + 1

    def _load_manifest(self) -> dict:
        if self.manifest_path.exists():
            return json.loads(self.manifest_path.read_text())
        return {'checkpoints': [], 'current_version': 0}

    def _save_manifest(self, manifest: dict):
        self.manifest_path.write_text(json.dumps(manifest, indent=2))
