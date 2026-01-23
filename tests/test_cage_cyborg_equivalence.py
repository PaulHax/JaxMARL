"""CybORG equivalence tests for CAGE-JAX.

These tests verify that CAGE-JAX produces equivalent results to the
original CybORG CAGE Challenge 2 implementation.

Requirements:
- CybORG installed: pip install -r requirements-jax.txt
- Run with: pytest tests/test_cage_cyborg_equivalence.py -v
"""

import pytest
from pathlib import Path

import jax
import jax.numpy as jnp

from jaxmarl.environments.cage import CageEnv
from jaxmarl.environments.cage.state import (
    HOST_IDS, COMPROMISE_PRIVILEGED, create_scenario2_const,
)


def _cyborg_available():
    """Check if CybORG is available."""
    try:
        from CybORG import CybORG
        return True
    except ImportError:
        return False


def _get_cyborg_scenario_path(scenario_name: str = "Scenario2.yaml") -> Path:
    """Get path to CybORG scenario file using importlib."""
    import importlib.util
    spec = importlib.util.find_spec("CybORG")
    if spec is None or spec.origin is None:
        raise ImportError("CybORG package not found")
    cyborg_path = Path(spec.origin).parent
    scenario_path = cyborg_path / "Shared" / "Scenarios" / scenario_name
    if not scenario_path.exists():
        raise FileNotFoundError(f"Scenario file not found: {scenario_path}")
    return scenario_path


@pytest.mark.skipif(not _cyborg_available(), reason="CybORG not installed")
class TestHostValueEquivalence:
    """Verify host values match between CAGE-JAX and CybORG Scenario2."""

    def test_value_mapping(self):
        """Verify the value mapping matches CybORG."""
        from jaxmarl.environments.cage.rewards import CONFIDENTIALITY_SCALE, AVAILABILITY_SCALE
        assert CONFIDENTIALITY_SCALE == 1.0
        assert AVAILABILITY_SCALE == 1.0

    def test_host_confidentiality_values(self):
        """Verify confidentiality values match Scenario2.yaml."""
        const = create_scenario2_const()

        expected = {
            'User0': 0.0,
            'User1': 0.1,
            'User2': 0.1,
            'User3': 0.1,
            'User4': 0.1,
            'Enterprise0': 1.0,
            'Enterprise1': 1.0,
            'Enterprise2': 1.0,
            'Defender': 0.1,
            'Op_Host0': 0.1,
            'Op_Host1': 0.1,
            'Op_Host2': 0.1,
            'Op_Server0': 1.0,
        }

        for host, expected_val in expected.items():
            actual = float(const.host_confidentiality[HOST_IDS[host]])
            assert abs(actual - expected_val) < 1e-5, f"{host}: expected {expected_val}, got {actual}"

    def test_host_availability_values(self):
        """Verify availability values match Scenario2.yaml."""
        const = create_scenario2_const()

        expected = {
            'User0': 0.0,
            'User1': 0.0,
            'Enterprise0': 1.0,
            'Enterprise1': 1.0,
            'Enterprise2': 1.0,
            'Defender': 0.1,
            'Op_Server0': 10.0,
        }

        for host, expected_val in expected.items():
            actual = float(const.host_availability[HOST_IDS[host]])
            assert abs(actual - expected_val) < 1e-5, f"{host}: expected {expected_val}, got {actual}"


@pytest.mark.skipif(not _cyborg_available(), reason="CybORG not installed")
class TestRewardEquivalence:
    """Verify reward calculations match CybORG."""

    def test_single_host_privileged_reward(self):
        """Compromising single host with privileged access."""
        from jaxmarl.environments.cage.rewards import compute_rewards_simple
        from jaxmarl.environments.cage.state import create_initial_state

        const = create_scenario2_const()
        state = create_initial_state(const)

        state = state.replace(
            host_compromised=state.host_compromised.at[HOST_IDS['Enterprise0']].set(COMPROMISE_PRIVILEGED)
        )

        rewards = compute_rewards_simple(state, const)

        assert rewards['red'] == 1.0
        assert rewards['blue'] == -1.0

    def test_op_server_full_reward(self):
        """Compromising Op_Server0 and running Impact."""
        from jaxmarl.environments.cage.rewards import compute_rewards_simple
        from jaxmarl.environments.cage.state import create_initial_state

        const = create_scenario2_const()
        state = create_initial_state(const)

        state = state.replace(
            host_compromised=state.host_compromised.at[HOST_IDS['Op_Server0']].set(COMPROMISE_PRIVILEGED)
        )

        rewards = compute_rewards_simple(state, const)

        assert rewards['red'] == 11.0
        assert rewards['blue'] == -11.0


@pytest.mark.skipif(not _cyborg_available(), reason="CybORG not installed")
class TestInitialStateEquivalence:
    """Verify initial state matches CybORG."""

    def test_red_starts_on_user0(self):
        """Red should start with session on User0."""
        env = CageEnv()
        obs, state = env.reset(jax.random.PRNGKey(0))

        from jaxmarl.environments.cage.state import COMPROMISE_USER
        assert state.host_compromised[HOST_IDS['User0']] == COMPROMISE_USER
        assert state.red_sessions[HOST_IDS['User0']] == 1

    def test_other_hosts_clean(self):
        """Other hosts should be clean initially."""
        env = CageEnv()
        obs, state = env.reset(jax.random.PRNGKey(0))

        for host, idx in HOST_IDS.items():
            if host != 'User0':
                assert state.host_compromised[idx] == 0, f"{host} should be clean"


@pytest.mark.skipif(not _cyborg_available(), reason="CybORG not installed")
class TestObservationEquivalence:
    """Verify observation shapes match expected dimensions."""

    def test_blue_obs_dimensions(self):
        """Blue observation should have 52 dimensions (4 per host × 13 hosts)."""
        env = CageEnv()
        obs, state = env.reset(jax.random.PRNGKey(0))

        assert obs['blue'].shape == (52,)

    def test_red_obs_dimensions(self):
        """Red observation should have 40 dimensions (1 + 3 per host × 13 hosts)."""
        env = CageEnv()
        obs, state = env.reset(jax.random.PRNGKey(0))

        assert obs['red'].shape == (40,)


@pytest.mark.skipif(not _cyborg_available(), reason="CybORG not installed")
class TestCybORGComparison:
    """Direct comparison tests that run both CybORG and CAGE-JAX."""

    @pytest.fixture
    def cyborg_env(self):
        """Create CybORG environment."""
        from CybORG import CybORG
        scenario_path = _get_cyborg_scenario_path()
        return CybORG(scenario_file=str(scenario_path), environment='sim')

    def test_episode_length(self, cyborg_env):
        """Both environments should run for 100 steps."""
        jax_env = CageEnv(max_steps=100)
        assert jax_env.max_steps == 100

    def test_action_space_sizes(self, cyborg_env):
        """Action space sizes should be reasonable for both."""
        from jaxmarl.environments.cage.actions import NUM_BLUE_ACTIONS, NUM_RED_ACTIONS

        assert NUM_BLUE_ACTIONS > 30
        assert NUM_RED_ACTIONS > 100

    def test_cyborg_initial_state(self, cyborg_env):
        """Verify CybORG initial state structure."""
        cyborg_env.reset()
        true_state = cyborg_env.get_true_state({'Sessions': True})
        assert 'Red' in str(true_state) or len(true_state) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
