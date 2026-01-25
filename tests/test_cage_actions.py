"""Tests for CAGE-JAX action implementation."""

import jax
import jax.numpy as jnp
import pytest

from jaxmarl.environments.cage.state import (
    CageState, CageConst,
    create_scenario2_const, create_initial_state, create_initial_state_with_red_foothold,
    HOST_IDS, COMPROMISE_NONE, COMPROMISE_USER, COMPROMISE_PRIVILEGED,
)
from jaxmarl.environments.cage.actions import (
    apply_blue_action, apply_red_action,
    get_blue_action_mask, get_red_action_mask,
    decode_blue_action, decode_red_action,
    BLUE_SLEEP, BLUE_MONITOR, BLUE_ANALYSE_START, BLUE_REMOVE_START, BLUE_RESTORE_START,
    BLUE_DECOY_START, NUM_BLUE_ACTIONS,
    RED_SLEEP, RED_DISCOVER_SUBNET_START, RED_SCAN_HOST_START, RED_EXPLOIT_START,
    RED_PRIVESC_START, RED_IMPACT_START, NUM_RED_ACTIONS,
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


class TestBlueActionDecoding:
    def test_decode_sleep(self, const):
        action_type, target, decoy = decode_blue_action(BLUE_SLEEP, const)
        assert action_type == 0

    def test_decode_monitor(self, const):
        action_type, target, decoy = decode_blue_action(BLUE_MONITOR, const)
        assert action_type == 1

    def test_decode_remove(self, const):
        action_type, target, decoy = decode_blue_action(BLUE_REMOVE_START + 5, const)
        assert action_type == 3
        assert target == 5

    def test_decode_restore(self, const):
        action_type, target, decoy = decode_blue_action(BLUE_RESTORE_START + 8, const)
        assert action_type == 4
        assert target == 8


class TestRedActionDecoding:
    def test_decode_sleep(self, const):
        action_type, subnet, host, exploit = decode_red_action(RED_SLEEP, const)
        assert action_type == 0

    def test_decode_discover(self, const):
        action_type, subnet, host, exploit = decode_red_action(RED_DISCOVER_SUBNET_START + 1, const)
        assert action_type == 1
        assert subnet == 1

    def test_decode_scan(self, const):
        action_type, subnet, host, exploit = decode_red_action(RED_SCAN_HOST_START + 5, const)
        assert action_type == 2
        assert host == 5

    def test_decode_privesc(self, const):
        action_type, subnet, host, exploit = decode_red_action(RED_PRIVESC_START + 3, const)
        assert action_type == 4
        assert host == 3


class TestBlueActions:
    def test_remove_clears_compromise(self, const, foothold_state):
        """Test that Remove action clears compromise on target host when activity detected."""
        # Set up: host is compromised at USER level AND activity has been detected
        state = foothold_state.replace(
            host_compromised=foothold_state.host_compromised.at[HOST_IDS['Enterprise0']].set(COMPROMISE_USER),
            red_sessions=foothold_state.red_sessions.at[HOST_IDS['Enterprise0']].set(1),
            red_privilege=foothold_state.red_privilege.at[HOST_IDS['Enterprise0']].set(COMPROMISE_USER),
            host_activity_detected=foothold_state.host_activity_detected.at[HOST_IDS['Enterprise0']].set(True),
        )

        remove_action = BLUE_REMOVE_START + HOST_IDS['Enterprise0']
        new_state = apply_blue_action(state, jnp.array(remove_action), const)

        assert new_state.host_compromised[HOST_IDS['Enterprise0']] == COMPROMISE_NONE
        assert new_state.red_sessions[HOST_IDS['Enterprise0']] == 0
        assert new_state.red_privilege[HOST_IDS['Enterprise0']] == COMPROMISE_NONE
        # After Remove, observation should be Unknown
        assert new_state.host_observation_unknown[HOST_IDS['Enterprise0']]

    def test_remove_requires_activity_detected(self, const, foothold_state):
        """Test that Remove does nothing if activity was not detected."""
        state = foothold_state.replace(
            host_compromised=foothold_state.host_compromised.at[HOST_IDS['Enterprise0']].set(COMPROMISE_USER),
            red_sessions=foothold_state.red_sessions.at[HOST_IDS['Enterprise0']].set(1),
            red_privilege=foothold_state.red_privilege.at[HOST_IDS['Enterprise0']].set(COMPROMISE_USER),
            host_activity_detected=foothold_state.host_activity_detected.at[HOST_IDS['Enterprise0']].set(False),
        )

        remove_action = BLUE_REMOVE_START + HOST_IDS['Enterprise0']
        new_state = apply_blue_action(state, jnp.array(remove_action), const)

        # Remove should NOT clear compromise if activity not detected
        assert new_state.host_compromised[HOST_IDS['Enterprise0']] == COMPROMISE_USER
        assert new_state.red_sessions[HOST_IDS['Enterprise0']] == 1

    def test_remove_cannot_remove_privileged(self, const, foothold_state):
        """Test that Remove cannot clear privileged (root/SYSTEM) access."""
        state = foothold_state.replace(
            host_compromised=foothold_state.host_compromised.at[HOST_IDS['Enterprise0']].set(COMPROMISE_PRIVILEGED),
            red_sessions=foothold_state.red_sessions.at[HOST_IDS['Enterprise0']].set(1),
            red_privilege=foothold_state.red_privilege.at[HOST_IDS['Enterprise0']].set(COMPROMISE_PRIVILEGED),
            host_activity_detected=foothold_state.host_activity_detected.at[HOST_IDS['Enterprise0']].set(True),
        )

        remove_action = BLUE_REMOVE_START + HOST_IDS['Enterprise0']
        new_state = apply_blue_action(state, jnp.array(remove_action), const)

        # Remove should NOT clear privileged access
        assert new_state.host_compromised[HOST_IDS['Enterprise0']] == COMPROMISE_PRIVILEGED
        assert new_state.red_privilege[HOST_IDS['Enterprise0']] == COMPROMISE_PRIVILEGED

    def test_restore_resets_host(self, const, foothold_state):
        """Test that Restore action resets host to initial state."""
        state = foothold_state.replace(
            host_compromised=foothold_state.host_compromised.at[HOST_IDS['Enterprise0']].set(COMPROMISE_PRIVILEGED),
            host_decoys=foothold_state.host_decoys.at[HOST_IDS['Enterprise0'], 0].set(True),
        )

        restore_action = BLUE_RESTORE_START + HOST_IDS['Enterprise0']
        new_state = apply_blue_action(state, jnp.array(restore_action), const)

        assert new_state.host_compromised[HOST_IDS['Enterprise0']] == COMPROMISE_NONE
        assert jnp.all(new_state.host_decoys[HOST_IDS['Enterprise0']] == False)

    def test_sleep_does_nothing(self, const, initial_state):
        """Test that Sleep action doesn't modify state."""
        new_state = apply_blue_action(initial_state, jnp.array(BLUE_SLEEP), const)

        assert jnp.array_equal(new_state.host_compromised, initial_state.host_compromised)
        assert jnp.array_equal(new_state.host_decoys, initial_state.host_decoys)


class TestRedActions:
    def test_discover_subnet(self, const, foothold_state):
        """Test DiscoverRemoteSystems reveals hosts in subnet."""
        key = jax.random.PRNGKey(42)

        discover_enterprise = RED_DISCOVER_SUBNET_START + 1  # Enterprise subnet
        new_state = apply_red_action(foothold_state, jnp.array(discover_enterprise), const, key)

        assert new_state.red_discovered_hosts[HOST_IDS['Enterprise0']]
        assert new_state.red_discovered_hosts[HOST_IDS['Enterprise1']]
        assert new_state.last_red_action_success

    def test_scan_host(self, const, foothold_state):
        """Test DiscoverNetworkServices scans a host."""
        key = jax.random.PRNGKey(42)

        state = foothold_state.replace(
            red_discovered_hosts=foothold_state.red_discovered_hosts.at[HOST_IDS['Enterprise0']].set(True)
        )

        scan_action = RED_SCAN_HOST_START + HOST_IDS['Enterprise0']
        new_state = apply_red_action(state, jnp.array(scan_action), const, key)

        assert new_state.red_scanned_hosts[HOST_IDS['Enterprise0']]
        assert new_state.last_red_action_success

    def test_scan_fails_if_not_discovered(self, const, foothold_state):
        """Test that scan fails if host not discovered."""
        key = jax.random.PRNGKey(42)

        scan_action = RED_SCAN_HOST_START + HOST_IDS['Enterprise0']
        new_state = apply_red_action(foothold_state, jnp.array(scan_action), const, key)

        assert not new_state.red_scanned_hosts[HOST_IDS['Enterprise0']]
        assert not new_state.last_red_action_success

    def test_exploit_success(self, const, foothold_state):
        """Test exploit action can compromise a host."""
        key = jax.random.PRNGKey(42)

        state = foothold_state.replace(
            red_discovered_hosts=foothold_state.red_discovered_hosts.at[HOST_IDS['Enterprise0']].set(True),
            red_scanned_hosts=foothold_state.red_scanned_hosts.at[HOST_IDS['Enterprise0']].set(True),
        )

        from jaxmarl.environments.cage.state import EXPLOIT_IDS, NUM_HOSTS
        exploit_action = RED_EXPLOIT_START + EXPLOIT_IDS['SSHBruteForce'] * NUM_HOSTS + HOST_IDS['Enterprise0']

        successes = 0
        for i in range(10):
            key, subkey = jax.random.split(key)
            new_state = apply_red_action(state, jnp.array(exploit_action), const, subkey)
            if new_state.host_compromised[HOST_IDS['Enterprise0']] == COMPROMISE_USER:
                successes += 1

        assert successes >= 5

    def test_privesc(self, const, foothold_state):
        """Test privilege escalation."""
        key = jax.random.PRNGKey(42)

        state = foothold_state.replace(
            host_compromised=foothold_state.host_compromised.at[HOST_IDS['Enterprise0']].set(COMPROMISE_USER),
            red_sessions=foothold_state.red_sessions.at[HOST_IDS['Enterprise0']].set(1),
            red_privilege=foothold_state.red_privilege.at[HOST_IDS['Enterprise0']].set(COMPROMISE_USER),
        )

        privesc_action = RED_PRIVESC_START + HOST_IDS['Enterprise0']

        successes = 0
        for i in range(10):
            key, subkey = jax.random.split(key)
            new_state = apply_red_action(state, jnp.array(privesc_action), const, subkey)
            if new_state.red_privilege[HOST_IDS['Enterprise0']] == COMPROMISE_PRIVILEGED:
                successes += 1

        assert successes >= 7


class TestActionMasks:
    def test_blue_remove_mask(self, const, initial_state, foothold_state):
        """Test Remove action mask requires activity detected AND user-level access.

        Note: Remove cannot clear privileged access. The initial foothold on User0
        has PRIVILEGED access (matching CybORG), so Remove won't work there.
        We test with a host that has USER-level access instead.
        """
        mask_initial = get_blue_action_mask(initial_state, const)

        # Foothold state: User0 has PRIVILEGED access (Remove can't clear this)
        # Set up Enterprise0 with USER-level access for testing Remove
        state_user_access = foothold_state.replace(
            host_compromised=foothold_state.host_compromised.at[HOST_IDS['Enterprise0']].set(COMPROMISE_USER),
            red_sessions=foothold_state.red_sessions.at[HOST_IDS['Enterprise0']].set(1),
            red_privilege=foothold_state.red_privilege.at[HOST_IDS['Enterprise0']].set(COMPROMISE_USER),
        )
        mask_no_detect = get_blue_action_mask(state_user_access, const)
        # Remove should NOT be valid without activity detection
        assert not mask_no_detect[BLUE_REMOVE_START + HOST_IDS['Enterprise0']]

        # Set up state with activity detected AND user-level access
        state_detected = state_user_access.replace(
            host_activity_detected=state_user_access.host_activity_detected.at[HOST_IDS['Enterprise0']].set(True),
        )
        mask_detected = get_blue_action_mask(state_detected, const)
        # Now Remove should be valid for Enterprise0 (user-level access + detected)
        assert mask_detected[BLUE_REMOVE_START + HOST_IDS['Enterprise0']]
        # User0 has PRIVILEGED access - Remove is NOT valid even with detection
        state_user0_detected = state_detected.replace(
            host_activity_detected=state_detected.host_activity_detected.at[HOST_IDS['User0']].set(True),
        )
        mask_user0 = get_blue_action_mask(state_user0_detected, const)
        assert not mask_user0[BLUE_REMOVE_START + HOST_IDS['User0']]

    def test_red_discover_mask(self, const, foothold_state):
        """Test DiscoverRemoteSystems mask based on Red's reach."""
        mask = get_red_action_mask(foothold_state, const)

        assert mask[RED_DISCOVER_SUBNET_START + 0]  # User subnet
        assert mask[RED_DISCOVER_SUBNET_START + 1]  # Enterprise subnet
        assert not mask[RED_DISCOVER_SUBNET_START + 2]

    def test_red_scan_mask(self, const, foothold_state):
        """Test scan mask based on discovered hosts."""
        mask = get_red_action_mask(foothold_state, const)

        assert mask[RED_SCAN_HOST_START + HOST_IDS['User0']]
        assert not mask[RED_SCAN_HOST_START + HOST_IDS['Enterprise0']]


class TestJITCompilation:
    def test_blue_action_jit(self, const, initial_state):
        """Test blue action can be JIT compiled."""
        @jax.jit
        def apply_and_check(state, action):
            new_state = apply_blue_action(state, action, const)
            return new_state.time

        result = apply_and_check(initial_state, jnp.array(BLUE_SLEEP))
        assert result == 0

    def test_red_action_jit(self, const, foothold_state):
        """Test red action can be JIT compiled."""
        @jax.jit
        def apply_and_check(state, action, key):
            new_state = apply_red_action(state, action, const, key)
            return new_state.last_red_action_success

        key = jax.random.PRNGKey(0)
        result = apply_and_check(foothold_state, jnp.array(RED_SLEEP), key)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
