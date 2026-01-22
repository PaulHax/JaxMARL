"""Pytest configuration for CAGE-JAX tests."""

import sys
from pathlib import Path

# Add CybORG to path for equivalence tests
CYBORG_PATH = Path('/home/paulhax/src/cyber/cage-challenge-2/CybORG')
if CYBORG_PATH.exists():
    sys.path.insert(0, str(CYBORG_PATH))
