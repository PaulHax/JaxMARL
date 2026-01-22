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
from jaxmarl.environments.cage.actions import BLUE_SLEEP, BLUE_RESTORE_START


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

    def test_user_compromise_half_value(self, const, initial_state):
        """User-level compromise gives 50% of host value."""
        # Compromise Enterprise0 (value = 1.0)
        state = initial_state.replace(
            host_compromised=initial_state.host_compromised.at[HOST_IDS['Enterprise0']].set(COMPROMISE_USER)
        )

        rewards = compute_rewards_simple(state, const)

        assert rewards['red'] == 0.5 * CONFIDENTIALITY_SCALE
        assert rewards['blue'] == -rewards['red']

    def test_privileged_compromise_full_value(self, const, initial_state):
        """Privileged compromise gives full host value."""
        # Compromise Enterprise0 with privileged access
        state = initial_state.replace(
            host_compromised=initial_state.host_compromised.at[HOST_IDS['Enterprise0']].set(COMPROMISE_PRIVILEGED)
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
            host_compromised=state.host_compromised.at[HOST_IDS['Enterprise1']].set(COMPROMISE_PRIVILEGED)
        )

        rewards = compute_rewards_simple(state, const)

        assert jnp.isclose(rewards['blue'] + rewards['red'], 0.0)


class TestOpServerReward:
    def test_op_server_availability_reward(self, const, initial_state):
        """Compromising Op_Server0 with privileged access gives availability reward."""
        state = initial_state.replace(
            host_compromised=initial_state.host_compromised.at[HOST_IDS['Op_Server0']].set(COMPROMISE_PRIVILEGED)
        )

        rewards = compute_rewards_simple(state, const)

        # Op_Server0 has confidentiality 10.0 and availability 10.0
        expected_red = 10.0 * CONFIDENTIALITY_SCALE + 10.0 * AVAILABILITY_SCALE
        assert rewards['red'] == expected_red

    def test_op_server_user_no_availability(self, const, initial_state):
        """User-level access on Op_Server0 doesn't trigger availability."""
        state = initial_state.replace(
            host_compromised=initial_state.host_compromised.at[HOST_IDS['Op_Server0']].set(COMPROMISE_USER)
        )

        rewards = compute_rewards_simple(state, const)

        # Only 50% confidentiality, no availability
        expected_red = 10.0 * 0.5 * CONFIDENTIALITY_SCALE
        assert rewards['red'] == expected_red


class TestActionCosts:
    def test_restore_action_cost(self, const, initial_state):
        """Restore action incurs cost for Blue."""
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


class TestMaxReward:
    def test_max_red_reward(self, const):
        """Calculate maximum possible Red reward."""
        max_reward = get_max_red_reward(const)

        # All hosts compromised: sum of all confidentiality + Op_Server0 availability
        # User hosts: 0, Enterprise: 3*1.0, Op: 3*1.0 + 10.0 = 16.0
        # Plus availability: 10.0
        expected = 16.0 + 10.0
        assert max_reward == expected


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
        """Foothold state (User0 with user access) should have zero reward."""
        rewards = compute_rewards_simple(foothold_state, const)

        # User0 has no confidentiality value
        assert rewards['red'] == 0.0

    def test_full_compromise_scenario(self, const, initial_state):
        """Test reward for realistic attack path: Enterprise + Op_Server."""
        # Red compromises Enterprise0 (user) and Op_Server0 (privileged)
        state = initial_state.replace(
            host_compromised=initial_state.host_compromised.at[HOST_IDS['Enterprise0']].set(COMPROMISE_USER)
        )
        state = state.replace(
            host_compromised=state.host_compromised.at[HOST_IDS['Op_Server0']].set(COMPROMISE_PRIVILEGED)
        )

        rewards = compute_rewards_simple(state, const)

        # Enterprise0: 0.5 * 1.0 = 0.5
        # Op_Server0: 1.0 * 10.0 + 10.0 (availability) = 20.0
        expected_red = 0.5 + 20.0
        assert rewards['red'] == expected_red


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
