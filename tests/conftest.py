import pytest

import omnigibson as og


def pytest_unconfigure(config):
    og.shutdown()


@pytest.fixture(params=["cpu", "cuda"])
def pipeline_mode(request):
    return request.param
