import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture(scope="session")
def spark():
    # Test data frames are created from pandas via Arrow. This avoids starting
    # Spark Python worker processes, which crash on Windows with Python 3.13.
    from tm.spark_utils import get_spark

    session = get_spark("tests", driver_memory="1g")
    yield session
    session.stop()
