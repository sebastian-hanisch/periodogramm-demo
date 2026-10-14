"""Gemeinsame Fixtures: die Mini-Instanz, an der jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def mini():
    """N = 8, fs = 1, x = cos(pi n / 2) = (1, 0, -1, 0, 1, 0, -1, 0).

    Von Hand: X_2 = X_6 = 4, alle anderen 0; I_2 = |X_2|^2 / (N fs) = 16 / 8 = 2; Parseval: sum |X|^2 = 32 = 8 * sum x^2 = 8 * 4.
    """
    return np.cos(np.pi * np.arange(8) / 2.0)
