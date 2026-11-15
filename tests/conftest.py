import numpy as np
import pytest

# The PDF's default case: short a 3-month at-the-money call.
S0, K, T, R, SIGMA = 100.0, 100.0, 0.25, 0.04, 0.20


@pytest.fixture
def rng():
    return np.random.default_rng(42)
