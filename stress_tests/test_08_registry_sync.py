import hashlib
from unittest.mock import patch, MagicMock
from core.registry_sync import GitHubRegistrySync

def test_registry_sync_rejects_tampered_algorithms():
    from core.registry_manager import DynamicRegistry
    from core.algorithm_embedder import AlgorithmEmbedder

    registry = DynamicRegistry(None)
    embedder = AlgorithmEmbedder()
    sync = GitHubRegistrySync(registry, embedder)

    original_code = """
class TamperedAlgo:
    def process(self, data):
        return sorted(data)
"""
    tampered_code = """
class TamperedAlgo:
    def process(self, data):
        import os
        os.system('rm -rf /')
        return sorted(data)
"""

    real_checksum = "sha256:" + hashlib.sha256(
        original_code.encode()
    ).hexdigest()

    mock_index = {
        "version": 99,
        "algorithms": [{
            "name": "TamperedAlgo",
            "version": "1.0.0",
            "status": "beta",
            "path": "algorithms/discovered/tampered",
            "checksum": real_checksum
        }]
    }

    mock_metadata = {
        "name": "TamperedAlgo",
        "problem_types": ["SORTING"],
        "embedding": [0.0] * 32
    }

    with patch.object(sync, '_fetch_index', return_value=mock_index), \
         patch('requests.get') as mock_get:

        def side_effect(url, **kwargs):
            resp = MagicMock()
            if url.endswith('.py'):
                resp.text = tampered_code
            else:
                resp.json.return_value = mock_metadata
            resp.raise_for_status = lambda: None
            return resp

        mock_get.side_effect = side_effect
        sync._save_state({})

        result = sync.sync_pull()

    registered = registry.list_algorithms()
    assert "TamperedAlgo" not in registered, \
        "FAIL: Tampered algorithm was registered despite checksum mismatch"

    print(f"\nRegistry Integrity Test:")
    print(f"  Tampered algorithm registered: "
          f"{'Yes' if 'TamperedAlgo' in registered else 'No'}")
    print(f"  Sync result: {result}")
