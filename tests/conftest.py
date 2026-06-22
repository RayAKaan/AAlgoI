"""pytest configuration and shared fixtures."""
import pytest
import tempfile


@pytest.fixture
def tmp_path():
    """Override tmp_path to avoid symlink creation issues on Windows."""
    path = tempfile.mkdtemp()
    import pathlib
    yield pathlib.Path(path)
