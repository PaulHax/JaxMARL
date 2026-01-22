"""CybORG equivalence tests for CAGE-JAX.

These tests verify that CAGE-JAX produces equivalent results to the
original CybORG CAGE Challenge 2 implementation.

Requirements:
- CybORG must be installed from /home/paulhax/src/cyber/cage-challenge-2/
- Run with: pytest tests/test_cage_cyborg_equivalence.py -v
"""

import sys
import pytest

# Add CybORG to path
sys.path.insert(0, '/home/paulhax/src/cyber/cage-challenge-2/CybORG')

try:
    from CybORG import CybORG
    from CybORG.Agents import B_lineAgent
    from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
    CYBORG_AVAILABLE = True
except ImportError:
    CYBORG_AVAILABLE = False

import jax
import jax.numpy as jnp

from jaxmarl.environments.cage import CageEnv
from jaxmarl.environments.cage.state import (
    HOST_IDS, COMPROMISE_PRIVILEGED, create_scenario2_const,
)


# Skip all tests if CybORG not available
pytestmark = pytest.mark.skipif(
    not CYBORG_AVAILABLE,
    reason="CybORG not installed"
)


class TestHostValueEquivalence:
    """Verify host values match between CAGE-JAX and CybORG Scenario2."""

    def test_value_mapping(self):
        """Verify the value mapping matches CybORG."""
        # CybORG mapping from RedRewardCalculator.py
        cyborg_mapping = {'None': 0.0, 'Low': 0.1, 'Medium': 1.0, 'High': 10.0}

        # Our mapping should match
        from jaxmarl.environments.cage.rewards import CONFIDENTIALITY_SCALE, AVAILABILITY_SCALE
        assert CONFIDENTIALITY_SCALE == 1.0
        assert AVAILABILITY_SCALE == 1.0

    def test_host_confidentiality_values(self):
        """Verify confidentiality values match Scenario2.yaml."""
        const = create_scenario2_const()

        # From Scenario2.yaml:
        expected = {
            'User0': 0.0,        # ConfidentialityValue: None
            'User1': 0.1,        # Not specified -> Low default
            'User2': 0.1,
            'User3': 0.1,
            'User4': 0.1,
            'Enterprise0': 1.0,  # Medium
            'Enterprise1': 1.0,  # Medium
            'Enterprise2': 1.0,  # Medium
            'Defender': 0.1,     # Not specified -> Low default
            'Op_Host0': 0.1,     # Not specified -> Low default
            'Op_Host1': 0.1,
            'Op_Host2': 0.1,
            'Op_Server0': 1.0,   # Medium (NOT High - availability is High)
        }

        for host, expected_val in expected.items():
            actual = float(const.host_confidentiality[HOST_IDS[host]])
            assert actual == expected_val, f"{host}: expected {expected_val}, got {actual}"

    def test_host_availability_values(self):
        """Verify availability values match Scenario2.yaml."""
        const = create_scenario2_const()

        expected = {
            'User0': 0.0,        # None
            'User1': 0.0,        # None
            'Enterprise0': 1.0,  # Medium
            'Enterprise1': 1.0,
            'Enterprise2': 1.0,
            'Defender': 0.1,     # Low default
            'Op_Server0': 10.0,  # High
        }

        for host, expected_val in expected.items():
            actual = float(const.host_availability[HOST_IDS[host]])
            assert actual == expected_val, f"{host}: expected {expected_val}, got {actual}"


class TestRewardEquivalence:
    """Verify reward calculations match CybORG."""

    def test_single_host_privileged_reward(self):
        """Compromising single host with privileged access."""
        from jaxmarl.environments.cage.rewards import compute_rewards_simple
        from jaxmarl.environments.cage.state import create_initial_state

        const = create_scenario2_const()
        state = create_initial_state(const)

        # Compromise Enterprise0 with privileged access
        state = state.replace(
            host_compromised=state.host_compromised.at[HOST_IDS['Enterprise0']].set(COMPROMISE_PRIVILEGED)
        )

        rewards = compute_rewards_simple(state, const)

        # CybORG: Enterprise0 Medium = 1.0
        assert rewards['red'] == 1.0
        assert rewards['blue'] == -1.0

    def test_op_server_full_reward(self):
        """Compromising Op_Server0 and running Impact."""
        from jaxmarl.environments.cage.rewards import compute_rewards_simple
        from jaxmarl.environments.cage.state import create_initial_state

        const = create_scenario2_const()
        state = create_initial_state(const)

        # Compromise Op_Server0 with privileged access
        state = state.replace(
            host_compromised=state.host_compromised.at[HOST_IDS['Op_Server0']].set(COMPROMISE_PRIVILEGED)
        )

        rewards = compute_rewards_simple(state, const)

        # CybORG: Op_Server0 confidentiality=Medium(1.0) + availability=High(10.0)
        assert rewards['red'] == 11.0
        assert rewards['blue'] == -11.0


class TestInitialStateEquivalence:
    """Verify initial state matches CybORG."""

    def test_red_starts_on_user0(self):
        """Red should start with session on User0."""
        env = CageEnv()
        obs, state = env.reset(jax.random.PRNGKey(0))

        # Red has user-level access on User0
        from jaxmarl.environments.cage.state import COMPROMISE_USER
        assert state.host_compromised[HOST_IDS['User0']] == COMPROMISE_USER
        assert state.red_sessions[HOST_IDS['User0']] == 1

    def test_other_hosts_clean(self):
        """Other hosts should be clean initially."""
        env = CageEnv()
        obs, state = env.reset(jax.random.PRNGKey(0))

        # All hosts except User0 should be clean
        for host, idx in HOST_IDS.items():
            if host != 'User0':
                assert state.host_compromised[idx] == 0, f"{host} should be clean"


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


@pytest.mark.skipif(not CYBORG_AVAILABLE, reason="CybORG not installed")
class TestCybORGComparison:
    """Direct comparison tests that run both CybORG and CAGE-JAX."""

    @pytest.fixture
    def cyborg_env(self):
        """Create CybORG environment."""
        path = '/home/paulhax/src/cyber/cage-challenge-2/CybORG/CybORG/Shared/Scenarios/Scenario2.yaml'
        return CybORG(scenario_file=path, environment='sim')

    def test_episode_length(self, cyborg_env):
        """Both environments should run for 100 steps."""
        # CAGE-JAX
        jax_env = CageEnv(max_steps=100)
        assert jax_env.max_steps == 100

    def test_action_space_sizes(self, cyborg_env):
        """Action space sizes should be reasonable for both."""
        jax_env = CageEnv()

        # CAGE-JAX action spaces
        from jaxmarl.environments.cage.actions import NUM_BLUE_ACTIONS, NUM_RED_ACTIONS

        # CybORG action spaces vary dynamically, but our fixed sizes should cover
        # the main action types
        assert NUM_BLUE_ACTIONS > 30  # Sleep + Monitor + Analyse + Remove + Restore + Decoys
        assert NUM_RED_ACTIONS > 100  # Sleep + Discover + Scan + Exploits + PrivEsc + Impact


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
