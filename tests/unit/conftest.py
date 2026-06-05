import pytest

from aalgoi.algorithms.base import Algorithm


class DummyAlgo(Algorithm):
    name = "dummy_algo"

    def process(self, data):
        return data


@pytest.fixture
def dummy_algo():
    return DummyAlgo()
