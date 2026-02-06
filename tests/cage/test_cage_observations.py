"""Tests for CAGE-JAX observation encoding."""

import jax
import jax.numpy as jnp
import pytest

from jaxmarl.environments.cage.state import (
    create_scenario2_const, create_initial_state, create_initial_state_with_red_foothold,
    HOST_IDS, COMPROMISE_USER, COMPROMISE_PRIVILEGED,
    ACTIVITY_SCAN, ACTIVITY_EXPLOIT,
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
    def test_blue_obs_shape(self, initial_state, const):
        obs = get_blue_obs(initial_state, const)
        assert obs.shape == (BLUE_OBS_DIM,)
        assert obs.shape == (52,)

    def test_red_obs_shape(self, initial_state, const):
        obs = get_red_obs(initial_state, const)
        assert obs.shape == (RED_OBS_DIM,)
        assert obs.shape == (40,)

    def test_get_obs_returns_both(self, initial_state, const):
        obs = get_obs(initial_state, const)
        assert 'blue' in obs
        assert 'red' in obs
        assert obs['blue'].shape == (52,)
        assert obs['red'].shape == (40,)


class TestBlueObservation:
    def test_initial_obs_zeros(self, initial_state, const):
        """Initial state should have mostly zero observations."""
        obs = get_blue_obs(initial_state, const)
        assert jnp.all(obs == 0.0)

    def test_compromised_host_visible(self, foothold_state, const):
        """Compromised host with Red session should be visible in blue observation.

        CybORG encoding: [activity_0, activity_1, compromised_0, compromised_1]
        For compromise with session and malware DETECTED by Blue:
        - compromised_0 = 1 (malware detected via Analyse)
        - compromised_1 = 1 (session detected)

        Note: Initial foothold (User0) is hidden; test Enterprise0 instead.
        Blue observes "privileged" via malware detection (Analyse), not actual privilege level.
        """
        state = foothold_state.replace(
            red_sessions=foothold_state.red_sessions.at[HOST_IDS['Enterprise0']].set(1),
            host_compromised=foothold_state.host_compromised.at[HOST_IDS['Enterprise0']].set(COMPROMISE_PRIVILEGED),
            host_has_malware=foothold_state.host_has_malware.at[HOST_IDS['Enterprise0']].set(True),
            host_malware_detected=foothold_state.host_malware_detected.at[HOST_IDS['Enterprise0']].set(True),
        )
        obs = get_blue_obs(state, const)

        ent0_idx = HOST_IDS['Enterprise0'] * BLUE_OBS_PER_HOST
        assert obs[ent0_idx + 2] == 1.0  # compromised_0: malware detected via Analyse
        assert obs[ent0_idx + 3] == 1.0  # compromised_1: session detected

    def test_privileged_access_visible(self, foothold_state, const):
        """Privileged access with malware DETECTED should set both compromise flags.

        CybORG encoding: compromised_0=1 (malware detected via Analyse), compromised_1=1 (session)
        Blue must use Analyse to detect malware; PrivEsc alone doesn't reveal Privileged.
        """
        state = foothold_state.replace(
            red_sessions=foothold_state.red_sessions.at[HOST_IDS['Enterprise0']].set(1),
            host_compromised=foothold_state.host_compromised.at[HOST_IDS['Enterprise0']].set(COMPROMISE_PRIVILEGED),
            host_has_malware=foothold_state.host_has_malware.at[HOST_IDS['Enterprise0']].set(True),
            host_malware_detected=foothold_state.host_malware_detected.at[HOST_IDS['Enterprise0']].set(True),
        )

        obs = get_blue_obs(state, const)

        ent0_idx = HOST_IDS['Enterprise0'] * BLUE_OBS_PER_HOST
        assert obs[ent0_idx + 2] == 1.0  # compromised_0: malware detected via Analyse
        assert obs[ent0_idx + 3] == 1.0  # compromised_1: session detected

    def test_detection_reflects_red_activity(self, foothold_state, const):
        """Activity features reflect red_activity_this_step (TRANSIENT per CybORG).

        CybORG activity is transient: only visible on the step the action occurs.
        Activity encoding: None=[0,0], Scan=[1,0], Exploit=[1,1]
        """
        # Test Scan activity: activity=[1,0]
        state = foothold_state.replace(
            red_activity_this_step=foothold_state.red_activity_this_step.at[HOST_IDS['Enterprise0']].set(ACTIVITY_SCAN),
        )

        obs = get_blue_obs(state, const)

        ent0_idx = HOST_IDS['Enterprise0'] * BLUE_OBS_PER_HOST
        assert obs[ent0_idx + 0] == 1.0  # activity_0: scan or exploit
        assert obs[ent0_idx + 1] == 0.0  # activity_1: exploit only

        # Test Exploit activity: activity=[1,1]
        state = foothold_state.replace(
            red_activity_this_step=foothold_state.red_activity_this_step.at[HOST_IDS['Enterprise0']].set(ACTIVITY_EXPLOIT),
        )

        obs = get_blue_obs(state, const)
        assert obs[ent0_idx + 0] == 1.0  # activity_0: scan or exploit
        assert obs[ent0_idx + 1] == 1.0  # activity_1: exploit


class TestRedObservation:
    def test_initial_obs_mostly_zeros(self, initial_state, const):
        """Initial state should have zero observations."""
        obs = get_red_obs(initial_state, const)
        assert jnp.all(obs == 0.0)

    def test_success_flag(self, foothold_state, const):
        """Success flag should reflect last action success."""
        state = foothold_state.replace(
            last_red_action_success=jnp.array(True)
        )

        obs = get_red_obs(state, const)
        assert obs[0] == 1.0  # success flag

    def test_foothold_visible(self, foothold_state, const):
        """Red's foothold should be visible in observation.

        CybORG starts Red with PRIVILEGED (SYSTEM) access on User0.
        CybORG encoding is mutually exclusive: None=[0,0], User=[1,0], Priv=[0,1]
        """
        obs = get_red_obs(foothold_state, const)

        user0_idx = 1 + HOST_IDS['User0'] * RED_OBS_PER_HOST
        assert obs[user0_idx + 0] == 1.0  # scanned
        assert obs[user0_idx + 1] == 0.0  # access_bit0: 0 for Privileged (not User)
        assert obs[user0_idx + 2] == 1.0  # access_bit1: 1 for Privileged

    def test_privileged_access_visible(self, foothold_state, const):
        """Privileged access should be visible with CybORG's mutually exclusive encoding.

        CybORG encoding: None=[0,0], User=[1,0], Priv=[0,1]
        """
        state = foothold_state.replace(
            red_privilege=foothold_state.red_privilege.at[HOST_IDS['User0']].set(COMPROMISE_PRIVILEGED)
        )

        obs = get_red_obs(state, const)

        user0_idx = 1 + HOST_IDS['User0'] * RED_OBS_PER_HOST
        assert obs[user0_idx + 1] == 0.0  # access_bit0: 0 for Privileged
        assert obs[user0_idx + 2] == 1.0  # access_bit1: 1 for Privileged


class TestJITCompilation:
    def test_blue_obs_jit(self, initial_state, const):
        @jax.jit
        def get_and_sum(state, const):
            obs = get_blue_obs(state, const)
            return jnp.sum(obs)

        result = get_and_sum(initial_state, const)
        assert result == 0.0

    def test_red_obs_jit(self, initial_state, const):
        @jax.jit
        def get_and_sum(state, const):
            obs = get_red_obs(state, const)
            return jnp.sum(obs)

        result = get_and_sum(initial_state, const)
        assert result == 0.0

    def test_combined_obs_jit(self, initial_state, const):
        @jax.jit
        def get_and_check(state, const):
            obs = get_obs(state, const)
            return obs['blue'].shape[0] + obs['red'].shape[0]

        result = get_and_check(initial_state, const)
        assert result == 92  # 52 + 40


class TestObservationNormalization:
    def test_blue_obs_bounded(self, foothold_state, const):
        """Blue observation values should be in [0, 1]."""
        state = foothold_state.replace(
            host_compromised=jnp.full_like(foothold_state.host_compromised, COMPROMISE_PRIVILEGED),
            red_sessions=jnp.ones_like(foothold_state.red_sessions),
            red_scanned_hosts_jax=jnp.ones_like(foothold_state.red_scanned_hosts_jax, dtype=jnp.bool_),
        )

        obs = get_blue_obs(state, const)
        assert jnp.all(obs >= 0.0)
        assert jnp.all(obs <= 1.0)

    def test_red_obs_bounded(self, foothold_state, const):
        """Red observation values should be in [0, 1]."""
        state = foothold_state.replace(
            last_red_action_success=jnp.array(True),
            red_scanned_hosts_jax=jnp.ones_like(foothold_state.red_scanned_hosts_jax, dtype=jnp.bool_),
            red_privilege=jnp.full_like(foothold_state.red_privilege, COMPROMISE_PRIVILEGED),
        )

        obs = get_red_obs(state, const)
        assert jnp.all(obs >= 0.0)
        assert jnp.all(obs <= 1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
