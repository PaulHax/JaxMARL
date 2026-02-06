"""Tests for CAGE-JAX reward calculation."""

import jax
import jax.numpy as jnp
import pytest

from jaxmarl.environments.cage.state import (
    create_scenario2_const, create_initial_state, create_initial_state_with_red_foothold,
    HOST_IDS, COMPROMISE_USER, COMPROMISE_PRIVILEGED, NUM_HOSTS,
)
from jaxmarl.environments.cage.rewards import (
    compute_rewards, compute_rewards_simple, get_max_red_reward,
    CONFIDENTIALITY_SCALE, AVAILABILITY_SCALE, BLUE_RESTORE_COST,
)
from jaxmarl.environments.cage.actions import (
    BLUE_SLEEP, BLUE_RESTORE_START, BLUE_REMOVE_START, BLUE_DECOY_START,
)


@pytest.fixture
def const():
    return create_scenario2_const()


@pytest.fixture
def initial_state(const):
    return create_initial_state(const)


@pytest.fixture
def foothold_state(const):
    return create_initial_state_with_red_foothold(const)


class TestBasicRewards:
    def test_zero_reward_no_compromise(self, const, initial_state):
        """No compromise means zero reward for both."""
        rewards = compute_rewards_simple(initial_state, const)

        assert rewards['blue'] == 0.0
        assert rewards['red'] == 0.0

    def test_user_compromise_no_reward(self, const, initial_state):
        """User-level compromise gives NO reward (CybORG only counts root/SYSTEM)."""
        # Compromise Enterprise0 with user access (value = 1.0)
        state = initial_state.replace(
            host_compromised=initial_state.host_compromised.at[HOST_IDS['Enterprise0']].set(COMPROMISE_USER)
        )

        rewards = compute_rewards_simple(state, const)

        # CybORG only gives reward for privileged (root/SYSTEM) sessions
        assert rewards['red'] == 0.0
        assert rewards['blue'] == 0.0

    def test_privileged_compromise_full_value(self, const, initial_state):
        """Privileged compromise gives full host confidentiality value."""
        # Compromise Enterprise0 with privileged access (value = 1.0)
        state = initial_state.replace(
            host_compromised=initial_state.host_compromised.at[HOST_IDS['Enterprise0']].set(COMPROMISE_PRIVILEGED),
            host_has_valid_privesc=initial_state.host_has_valid_privesc.at[HOST_IDS['Enterprise0']].set(True),
        )

        rewards = compute_rewards_simple(state, const)

        assert rewards['red'] == 1.0 * CONFIDENTIALITY_SCALE
        assert rewards['blue'] == -rewards['red']


class TestZeroSum:
    def test_zero_sum_simple(self, const, foothold_state):
        """Blue and Red rewards should be zero-sum (before action costs)."""
        rewards = compute_rewards_simple(foothold_state, const)

        assert jnp.isclose(rewards['blue'] + rewards['red'], 0.0)

    def test_zero_sum_with_multiple_compromises(self, const, initial_state):
        """Zero-sum holds with multiple compromised hosts."""
        state = initial_state.replace(
            host_compromised=initial_state.host_compromised.at[HOST_IDS['Enterprise0']].set(COMPROMISE_USER)
        )
        state = state.replace(
            host_compromised=state.host_compromised.at[HOST_IDS['Enterprise1']].set(COMPROMISE_PRIVILEGED),
            host_has_valid_privesc=state.host_has_valid_privesc.at[HOST_IDS['Enterprise1']].set(True),
        )

        rewards = compute_rewards_simple(state, const)

        assert jnp.isclose(rewards['blue'] + rewards['red'], 0.0)


class TestOpServerReward:
    def test_op_server_availability_reward(self, const, initial_state):
        """Availability reward requires Impact action to stop OT service."""
        # Privileged access alone doesn't give availability reward
        state = initial_state.replace(
            host_compromised=initial_state.host_compromised.at[HOST_IDS['Op_Server0']].set(COMPROMISE_PRIVILEGED),
            host_has_valid_privesc=initial_state.host_has_valid_privesc.at[HOST_IDS['Op_Server0']].set(True),
        )

        rewards = compute_rewards_simple(state, const)

        # Only confidentiality, no availability (Impact not run)
        expected_red = 1.0 * CONFIDENTIALITY_SCALE
        assert rewards['red'] == expected_red

    def test_op_server_with_impact(self, const, initial_state):
        """Availability reward given when Impact stops OT service."""
        state = initial_state.replace(
            host_compromised=initial_state.host_compromised.at[HOST_IDS['Op_Server0']].set(COMPROMISE_PRIVILEGED),
            host_has_valid_privesc=initial_state.host_has_valid_privesc.at[HOST_IDS['Op_Server0']].set(True),
            ot_service_stopped=initial_state.ot_service_stopped.at[HOST_IDS['Op_Server0']].set(True),
        )

        rewards = compute_rewards_simple(state, const)

        # Op_Server0: confidentiality=Medium(1.0), availability=High(10.0)
        expected_red = 1.0 * CONFIDENTIALITY_SCALE + 10.0 * AVAILABILITY_SCALE
        assert rewards['red'] == expected_red

    def test_op_server_user_no_reward(self, const, initial_state):
        """User-level access on Op_Server0 gives NO reward (need root/SYSTEM)."""
        state = initial_state.replace(
            host_compromised=initial_state.host_compromised.at[HOST_IDS['Op_Server0']].set(COMPROMISE_USER)
        )

        rewards = compute_rewards_simple(state, const)

        # CybORG: no reward for user-level sessions
        expected_red = 0.0
        assert rewards['red'] == expected_red


class TestActionCosts:
    def test_restore_action_cost(self, const, initial_state):
        """Restore action incurs cost for Blue (CybORG Restore.cost = -1)."""
        restore_action = jnp.array(BLUE_RESTORE_START + HOST_IDS['Enterprise0'])
        sleep_action = jnp.array(0)  # Red sleep

        rewards = compute_rewards(initial_state, const, restore_action, sleep_action)

        assert rewards['blue'] == BLUE_RESTORE_COST  # No compromise, just cost

    def test_no_cost_for_other_actions(self, const, initial_state):
        """Non-restore actions have no cost."""
        sleep_action = jnp.array(BLUE_SLEEP)
        red_sleep = jnp.array(0)

        rewards = compute_rewards(initial_state, const, sleep_action, red_sleep)

        assert rewards['blue'] == 0.0  # No compromise, no cost

    def test_remove_has_no_penalty(self, const, initial_state):
        """Remove actions have no penalty (CybORG Remove.cost = 0)."""
        state = initial_state.replace(last_blue_action_success=jnp.array(False))
        remove_action = jnp.array(BLUE_REMOVE_START + HOST_IDS['Enterprise0'])
        red_sleep = jnp.array(0)

        rewards = compute_rewards(state, const, remove_action, red_sleep)

        assert rewards['blue'] == 0.0  # No penalty for Remove (CybORG Remove.cost = 0)

    def test_successful_action_no_penalty(self, const, initial_state):
        """Successful actions don't incur invalid action penalty."""
        state = initial_state.replace(last_blue_action_success=jnp.array(True))
        sleep_action = jnp.array(BLUE_SLEEP)
        red_sleep = jnp.array(0)

        rewards = compute_rewards(state, const, sleep_action, red_sleep)

        assert rewards['blue'] == 0.0

    def test_failed_decoy_no_penalty(self, const, initial_state):
        """Failed Decoy actions do NOT incur penalty (CybORG Decoys fail silently)."""
        state = initial_state.replace(last_blue_action_success=jnp.array(False))
        decoy_action = jnp.array(BLUE_DECOY_START)  # First decoy action
        red_sleep = jnp.array(0)

        rewards = compute_rewards(state, const, decoy_action, red_sleep)

        assert rewards['blue'] == 0.0  # No penalty for failed decoy


class TestMaxReward:
    def test_max_red_reward(self, const):
        """Calculate maximum possible Red reward (CybORG Scenario2 values)."""
        max_reward = get_max_red_reward(const)

        # From Scenario2.yaml with CybORG value mapping:
        # User0: 0.0, User1-4: 0.1*4=0.4
        # Enterprise0-2: 1.0*3=3.0
        # Defender: 0.1
        # Op_Host0-2: 0.1*3=0.3
        # Op_Server0 confidentiality: 1.0
        # Op_Server0 availability: 10.0
        # Total: 0.0 + 0.4 + 3.0 + 0.1 + 0.3 + 1.0 + 10.0 = 14.8
        expected = 14.8
        assert abs(max_reward - expected) < 0.01


class TestJITCompilation:
    def test_rewards_jit(self, const, initial_state):
        """Reward computation should JIT compile."""
        @jax.jit
        def compute_and_sum(state):
            rewards = compute_rewards_simple(state, const)
            return rewards['blue'] + rewards['red']

        result = compute_and_sum(initial_state)
        assert result == 0.0

    def test_rewards_with_actions_jit(self, const, initial_state):
        """Reward computation with actions should JIT compile."""
        @jax.jit
        def compute(state, blue_action, red_action):
            rewards = compute_rewards(state, const, blue_action, red_action)
            return rewards['blue'], rewards['red']

        blue_r, red_r = compute(initial_state, jnp.array(0), jnp.array(0))
        assert blue_r == 0.0
        assert red_r == 0.0


class TestRewardScenarios:
    def test_foothold_reward(self, const, foothold_state):
        """Foothold state (User0 with USER access) should have zero reward."""
        rewards = compute_rewards_simple(foothold_state, const)

        # CybORG: user-level doesn't count, only root/SYSTEM
        assert rewards['red'] == 0.0

    def test_full_compromise_scenario(self, const, initial_state):
        """Test reward for realistic attack path: Enterprise + Op_Server with Impact."""
        # Red compromises Enterprise0 (privileged) and Op_Server0 (privileged + Impact)
        state = initial_state.replace(
            host_compromised=initial_state.host_compromised.at[HOST_IDS['Enterprise0']].set(COMPROMISE_PRIVILEGED),
            host_has_valid_privesc=initial_state.host_has_valid_privesc.at[HOST_IDS['Enterprise0']].set(True),
        )
        state = state.replace(
            host_compromised=state.host_compromised.at[HOST_IDS['Op_Server0']].set(COMPROMISE_PRIVILEGED),
            host_has_valid_privesc=state.host_has_valid_privesc.at[HOST_IDS['Op_Server0']].set(True),
            ot_service_stopped=state.ot_service_stopped.at[HOST_IDS['Op_Server0']].set(True),
        )

        rewards = compute_rewards_simple(state, const)

        # Enterprise0: 1.0 (Medium) confidentiality
        # Op_Server0: 1.0 (Medium) confidentiality + 10.0 (High) availability = 11.0
        expected_red = 1.0 + 11.0
        assert rewards['red'] == expected_red


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
