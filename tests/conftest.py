"""Pytest configuration for CAGE-JAX tests."""

import pytest
import jax
import jax.numpy as jnp


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "cyborg: tests that require CybORG to be installed"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )


@pytest.fixture
def cyborg_available():
    """Check if CybORG is available."""
    try:
        from CybORG import CybORG
        return True
    except ImportError:
        return False


@pytest.fixture(scope="session")
def scenario2_config():
    """Session-scoped Scenario2 config."""
    from jaxmarl.environments.cage.config import get_scenario
    return get_scenario('Scenario2')


@pytest.fixture(scope="session")
def scenario2_env(scenario2_config):
    """Session-scoped CageEnv for Scenario2 with pre-warmed JIT."""
    from jaxmarl.environments.cage import CageEnv
    from jaxmarl.environments.cage.actions import BLUE_SLEEP

    env = CageEnv(config=scenario2_config, max_steps=100)
    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key)
    actions = {'blue': jnp.array(BLUE_SLEEP), 'red': jnp.array(0)}
    env.step_env(key, state, actions)
    return env


@pytest.fixture(scope="session")
def jax_key():
    """Session-scoped base JAX random key."""
    return jax.random.PRNGKey(42)
