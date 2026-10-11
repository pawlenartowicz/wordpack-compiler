import pytest

import fixture


@pytest.fixture(scope="session")
def pack_path(tmp_path_factory):
    return fixture.build_fixture(tmp_path_factory.mktemp("pack"))


@pytest.fixture(scope="session")
def cased_pack_path(tmp_path_factory):
    return fixture.build_fixture(tmp_path_factory.mktemp("cased"), case_sensitive=True)
