"""Pytest configuration for CAGE-JAX tests."""

import pytest


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "cyborg: tests that require CybORG to be installed"
    )


@pytest.fixture
def cyborg_available():
    """Check if CybORG is available."""
    try:
        from CybORG import CybORG
        return True
    except ImportError:
        return False
