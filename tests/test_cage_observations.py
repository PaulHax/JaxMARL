"""Tests for CAGE-JAX observation encoding."""

import jax
import jax.numpy as jnp
import pytest

from jaxmarl.environments.cage.state import (
    create_scenario2_const, create_initial_state, create_initial_state_with_red_foothold,
    HOST_IDS, COMPROMISE_USER, COMPROMISE_PRIVILEGED,
)
from jaxmarl.environments.cage.observations import (
    get_blue_obs, get_red_obs, get_obs,
    BLUE_OBS_DIM, RED_OBS_DIM, BLUE_OBS_PER_HOST, RED_OBS_PER_HOST,
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


class TestObservationShapes:
    def test_blue_obs_shape(self, initial_state):
        obs = get_blue_obs(initial_state)
        assert obs.shape == (BLUE_OBS_DIM,)
        assert obs.shape == (52,)

    def test_red_obs_shape(self, initial_state):
        obs = get_red_obs(initial_state)
        assert obs.shape == (RED_OBS_DIM,)
        assert obs.shape == (40,)

    def test_get_obs_returns_both(self, initial_state):
        obs = get_obs(initial_state)
        assert 'blue' in obs
        assert 'red' in obs
        assert obs['blue'].shape == (52,)
        assert obs['red'].shape == (40,)


class TestBlueObservation:
    def test_initial_obs_zeros(self, initial_state):
        """Initial state should have mostly zero observations."""
        obs = get_blue_obs(initial_state)
        # No compromise, no detection
        assert jnp.all(obs == 0.0)

    def test_compromised_host_visible(self, foothold_state):
        """Compromised host should be visible in blue observation."""
        obs = get_blue_obs(foothold_state)

        # User0 is compromised with user-level access
        user0_idx = HOST_IDS['User0'] * BLUE_OBS_PER_HOST
        assert obs[user0_idx + 2] == 1.0  # user_compromised
        assert obs[user0_idx + 3] == 0.0  # not privileged

    def test_privileged_access_visible(self, foothold_state):
        """Privileged access should set both user and privileged flags."""
        state = foothold_state.replace(
            host_compromised=foothold_state.host_compromised.at[HOST_IDS['Enterprise0']].set(COMPROMISE_PRIVILEGED)
        )

        obs = get_blue_obs(state)

        ent0_idx = HOST_IDS['Enterprise0'] * BLUE_OBS_PER_HOST
        assert obs[ent0_idx + 2] == 1.0  # user_compromised (privileged implies user)
        assert obs[ent0_idx + 3] == 1.0  # privileged_compromised

    def test_detection_reflects_red_activity(self, foothold_state):
        """Detection features should reflect red activity."""
        # Set red session on Enterprise0
        state = foothold_state.replace(
            red_sessions=foothold_state.red_sessions.at[HOST_IDS['Enterprise0']].set(1),
            red_scanned_hosts=foothold_state.red_scanned_hosts.at[HOST_IDS['Enterprise0']].set(True),
        )

        obs = get_blue_obs(state)

        ent0_idx = HOST_IDS['Enterprise0'] * BLUE_OBS_PER_HOST
        assert obs[ent0_idx + 0] == 1.0  # scan_detected
        assert obs[ent0_idx + 1] == 1.0  # exploit_detected (session exists)


class TestRedObservation:
    def test_initial_obs_mostly_zeros(self, initial_state):
        """Initial state should have zero observations."""
        obs = get_red_obs(initial_state)
        assert jnp.all(obs == 0.0)

    def test_success_flag(self, foothold_state):
        """Success flag should reflect last action success."""
        state = foothold_state.replace(
            last_red_action_success=jnp.array(True)
        )

        obs = get_red_obs(state)
        assert obs[0] == 1.0  # success flag

    def test_foothold_visible(self, foothold_state):
        """Red's foothold should be visible in observation."""
        obs = get_red_obs(foothold_state)

        # User0 is scanned and has user access
        user0_idx = 1 + HOST_IDS['User0'] * RED_OBS_PER_HOST
        assert obs[user0_idx + 0] == 1.0  # scanned
        assert obs[user0_idx + 1] == 1.0  # user_access
        assert obs[user0_idx + 2] == 0.0  # no privileged_access

    def test_privileged_access_visible(self, foothold_state):
        """Privileged access should be visible."""
        state = foothold_state.replace(
            red_privilege=foothold_state.red_privilege.at[HOST_IDS['User0']].set(COMPROMISE_PRIVILEGED)
        )

        obs = get_red_obs(state)

        user0_idx = 1 + HOST_IDS['User0'] * RED_OBS_PER_HOST
        assert obs[user0_idx + 1] == 1.0  # user_access
        assert obs[user0_idx + 2] == 1.0  # privileged_access


class TestJITCompilation:
    def test_blue_obs_jit(self, initial_state):
        @jax.jit
        def get_and_sum(state):
            obs = get_blue_obs(state)
            return jnp.sum(obs)

        result = get_and_sum(initial_state)
        assert result == 0.0

    def test_red_obs_jit(self, initial_state):
        @jax.jit
        def get_and_sum(state):
            obs = get_red_obs(state)
            return jnp.sum(obs)

        result = get_and_sum(initial_state)
        assert result == 0.0

    def test_combined_obs_jit(self, initial_state):
        @jax.jit
        def get_and_check(state):
            obs = get_obs(state)
            return obs['blue'].shape[0] + obs['red'].shape[0]

        result = get_and_check(initial_state)
        assert result == 92  # 52 + 40


class TestObservationNormalization:
    def test_blue_obs_bounded(self, foothold_state):
        """Blue observation values should be in [0, 1]."""
        # Create worst-case state
        state = foothold_state.replace(
            host_compromised=jnp.full_like(foothold_state.host_compromised, COMPROMISE_PRIVILEGED),
            red_sessions=jnp.ones_like(foothold_state.red_sessions),
            red_scanned_hosts=jnp.ones_like(foothold_state.red_scanned_hosts, dtype=jnp.bool_),
        )

        obs = get_blue_obs(state)
        assert jnp.all(obs >= 0.0)
        assert jnp.all(obs <= 1.0)

    def test_red_obs_bounded(self, foothold_state):
        """Red observation values should be in [0, 1]."""
        state = foothold_state.replace(
            last_red_action_success=jnp.array(True),
            red_scanned_hosts=jnp.ones_like(foothold_state.red_scanned_hosts, dtype=jnp.bool_),
            red_privilege=jnp.full_like(foothold_state.red_privilege, COMPROMISE_PRIVILEGED),
        )

        obs = get_red_obs(state)
        assert jnp.all(obs >= 0.0)
        assert jnp.all(obs <= 1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
