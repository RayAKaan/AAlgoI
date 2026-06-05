import json

import pytest

from aalgoi.core.mind.federated_mind import FederatedMindSync, StructuralHasher


@pytest.fixture
def sync(tmp_path):
    sync_dir = tmp_path / "federation"
    return FederatedMindSync(str(sync_dir))


class TestStructuralHasher:
    def test_hash_is_deterministic(self):
        h = StructuralHasher()
        algo = {"name": "quick_sort", "type": "sorting"}
        assert h.hash_algorithm(algo) == h.hash_algorithm(algo)

    def test_hash_differs_for_different_algorithms(self):
        h = StructuralHasher()
        a1 = h.hash_algorithm({"name": "quick_sort"})
        a2 = h.hash_algorithm({"name": "merge_sort"})
        assert a1 != a2

    def test_hash_ignores_key_order(self):
        h = StructuralHasher()
        a1 = h.hash_algorithm({"a": 1, "b": 2})
        a2 = h.hash_algorithm({"b": 2, "a": 1})
        assert a1 == a2

    def test_signature_matches_hash(self):
        h = StructuralHasher()
        algo = {"name": "test"}
        assert h.compute_signature(algo) == h.hash_algorithm(algo)

    def test_handles_non_serializable(self):
        h = StructuralHasher()
        algo = {"name": "test", "complex": object()}
        result = h.hash_algorithm(algo)
        assert isinstance(result, str)
        assert len(result) == 64


class TestOutboxCreation:
    def test_outbox_directory_created(self, sync):
        assert sync.outbox_dir.is_dir()

    def test_inbox_directory_created(self, sync):
        assert sync.inbox_dir.is_dir()

    def test_sync_dir_created(self, sync):
        assert sync.sync_dir.is_dir()


class TestAnonymizedShare:
    def test_share_creates_file_in_outbox(self, sync):
        sync.anonymized_share({"name": "quick_sort"})
        files = list(sync.outbox_dir.glob("*.json"))
        assert len(files) == 1

    def test_share_payload_has_required_fields(self, sync):
        sync.anonymized_share({"name": "quick_sort"})
        fpath = list(sync.outbox_dir.glob("*.json"))[0]
        with open(fpath) as f:
            payload = json.load(f)
        assert "structural_hash" in payload
        assert "signature" in payload
        assert "epsilon" in payload
        assert payload["epsilon"] == 0.1

    def test_share_multiple_creates_separate_files(self, sync):
        sync.anonymized_share({"name": "a"})
        sync.anonymized_share({"name": "b"})
        assert len(list(sync.outbox_dir.glob("*.json"))) == 2


class TestSyncRoundtrip:
    def test_sync_returns_empty_when_no_messages(self, sync):
        assert sync.sync() == []

    def test_sync_reads_inbox_messages(self, sync):
        msg = {"structural_hash": "abc", "signature": "def", "epsilon": 0.1}
        fpath = sync.inbox_dir / "msg_001.json"
        with open(fpath, "w") as f:
            json.dump(msg, f)
        result = sync.sync()
        assert len(result) == 1
        assert result[0] == msg

    def test_sync_removes_processed_files(self, sync):
        msg = {"structural_hash": "abc", "signature": "def", "epsilon": 0.1}
        fpath = sync.inbox_dir / "msg_001.json"
        with open(fpath, "w") as f:
            json.dump(msg, f)
        sync.sync()
        assert not fpath.exists()

    def test_sync_messages_in_order(self, sync):
        for i in range(5):
            with open(sync.inbox_dir / f"msg_{i:03d}.json", "w") as f:
                json.dump({"id": i}, f)
        result = sync.sync(max_messages=10)
        assert [m["id"] for m in result] == [0, 1, 2, 3, 4]

    def test_share_and_sync_roundtrip(self, sync, tmp_path):
        sync.anonymized_share({"name": "bubble_sort"})
        outbox_file = list(sync.outbox_dir.glob("*.json"))[0]
        data = json.loads(outbox_file.read_text())
        inbox_file = sync.inbox_dir / outbox_file.name
        inbox_file.write_text(json.dumps(data))
        result = sync.sync()
        assert len(result) == 1
        assert result[0]["structural_hash"] == data["structural_hash"]


class TestMessageOrderAndWindow:
    def test_window_limits_messages(self, sync):
        for i in range(15):
            with open(sync.inbox_dir / f"msg_{i:03d}.json", "w") as f:
                json.dump({"id": i}, f)
        result = sync.sync(max_messages=10)
        assert len(result) == 10
        assert [m["id"] for m in result] == list(range(10))

    def test_excess_messages_remain_in_inbox(self, sync):
        for i in range(15):
            with open(sync.inbox_dir / f"msg_{i:03d}.json", "w") as f:
                json.dump({"id": i}, f)
        sync.sync(max_messages=10)
        remaining = len(list(sync.inbox_dir.glob("*.json")))
        assert remaining == 5

    def test_second_sync_gets_remaining(self, sync):
        for i in range(12):
            with open(sync.inbox_dir / f"msg_{i:03d}.json", "w") as f:
                json.dump({"id": i}, f)
        r1 = sync.sync(max_messages=10)
        assert len(r1) == 10
        r2 = sync.sync(max_messages=10)
        assert len(r2) == 2
        assert [m["id"] for m in r2] == [10, 11]

    def test_window_uses_default_max(self, sync):
        for i in range(15):
            with open(sync.inbox_dir / f"msg_{i:03d}.json", "w") as f:
                json.dump({"id": i}, f)
        result = sync.sync()
        assert len(result) == 10
