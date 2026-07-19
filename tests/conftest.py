"""Register POS benchmark options before the performance test is collected."""

from tests.pos import conftest as pos_harness


def pytest_addoption(parser):
    pos_harness.pytest_addoption(parser)


def pytest_configure(config):
    pos_harness.pytest_configure(config)
