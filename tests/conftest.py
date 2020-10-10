import pytest


@pytest.fixture(scope="session")
def data_files_dir(tmpdir_factory):
    datadir = tmpdir_factory.mktemp("data")
    # create_data_files(str(datadir))
    return datadir
