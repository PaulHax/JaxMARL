"""Tests for CAGE-JAX state dataclasses."""

import jax
import jax.numpy as jnp
import pytest

from jaxmarl.environments.cage.state import (
    CageState, CageConst,
    create_scenario2_const, create_initial_state, create_initial_state_with_red_foothold,
    NUM_HOSTS, NUM_SERVICES, NUM_DECOY_TYPES, MAX_PROCESSES, NUM_EXPLOITS,
    HOST_IDS, COMPROMISE_USER, COMPROMISE_PRIVILEGED, COMPROMISE_NONE,
)


def test_cageconst_creation():
    """Test CageConst is created with correct shapes."""
    const = create_scenario2_const()

    assert const.adjacency.shape == (NUM_HOSTS, NUM_HOSTS)
    assert const.subnet_adjacency.shape == (3, 3)
    assert const.host_os.shape == (NUM_HOSTS,)
    assert const.host_confidentiality.shape == (NUM_HOSTS,)
    assert const.host_availability.shape == (NUM_HOSTS,)
    assert const.initial_services.shape == (NUM_HOSTS, NUM_SERVICES)
    assert const.service_exploits.shape == (NUM_SERVICES, NUM_EXPLOITS)
    assert const.max_steps == 100
    assert const.num_hosts == NUM_HOSTS


def test_cagestate_creation():
    """Test CageState is created with correct shapes."""
    const = create_scenario2_const()
    state = create_initial_state(const)

    assert state.time == 0
    assert state.done == False
    assert state.host_compromised.shape == (NUM_HOSTS,)
    assert state.host_services.shape == (NUM_HOSTS, NUM_SERVICES)
    assert state.host_processes.shape == (NUM_HOSTS, MAX_PROCESSES)
    assert state.host_decoys.shape == (NUM_HOSTS, NUM_DECOY_TYPES)
    assert state.red_sessions.shape == (NUM_HOSTS,)
    assert state.red_privilege.shape == (NUM_HOSTS,)
    assert state.blue_sessions.shape == (NUM_HOSTS,)
    assert state.red_discovered_hosts.shape == (NUM_HOSTS,)
    assert state.red_scanned_hosts.shape == (NUM_HOSTS,)


def test_initial_state_values():
    """Test initial state has correct default values."""
    const = create_scenario2_const()
    state = create_initial_state(const)

    # No hosts compromised initially
    assert jnp.all(state.host_compromised == COMPROMISE_NONE)

    # No red sessions initially
    assert jnp.all(state.red_sessions == 0)

    # Blue has sessions on all hosts
    assert jnp.all(state.blue_sessions == 1)

    # Red hasn't discovered or scanned any hosts
    assert jnp.all(state.red_discovered_hosts == False)
    assert jnp.all(state.red_scanned_hosts == False)


def test_initial_state_with_foothold():
    """Test initial state with Red foothold on User0.

    CybORG starts Red with SYSTEM (PRIVILEGED) access on the foothold host.
    """
    const = create_scenario2_const()
    state = create_initial_state_with_red_foothold(const)

    # User0 should be compromised at PRIVILEGED level (matches CybORG)
    assert state.host_compromised[HOST_IDS['User0']] == COMPROMISE_PRIVILEGED
    assert state.red_privilege[HOST_IDS['User0']] == COMPROMISE_PRIVILEGED

    # Red should have session on User0
    assert state.red_sessions[HOST_IDS['User0']] == 1

    # Red should have discovered and scanned User0
    assert state.red_discovered_hosts[HOST_IDS['User0']] == True
    assert state.red_scanned_hosts[HOST_IDS['User0']] == True

    # Other hosts should be clean
    assert state.host_compromised[HOST_IDS['Enterprise0']] == COMPROMISE_NONE


def test_state_jit_compiles():
    """Verify state operations can be JIT compiled."""
    const = create_scenario2_const()

    @jax.jit
    def create_and_modify_state():
        state = create_initial_state(const)
        state = state.replace(time=state.time + 1)
        state = state.replace(
            host_compromised=state.host_compromised.at[0].set(COMPROMISE_USER)
        )
        return state

    state = create_and_modify_state()
    assert state.time == 1
    assert state.host_compromised[0] == COMPROMISE_USER


def test_subnet_adjacency():
    """Test subnet connectivity is configured correctly."""
    const = create_scenario2_const()

    # User hosts can reach Enterprise hosts
    assert const.adjacency[HOST_IDS['User0'], HOST_IDS['Enterprise0']]

    # User hosts cannot directly reach Operational hosts
    assert not const.adjacency[HOST_IDS['User0'], HOST_IDS['Op_Server0']]

    # Enterprise hosts can reach Operational hosts
    assert const.adjacency[HOST_IDS['Enterprise0'], HOST_IDS['Op_Server0']]


def test_confidentiality_values():
    """Test confidentiality values match CybORG Scenario 2."""
    const = create_scenario2_const()

    # User0 has None confidentiality (0.0)
    assert const.host_confidentiality[HOST_IDS['User0']] == 0.0

    # User1-4 default to Low (0.1) per CybORG
    assert const.host_confidentiality[HOST_IDS['User1']] == 0.1

    # Enterprise hosts have Medium value (1.0)
    assert const.host_confidentiality[HOST_IDS['Enterprise0']] == 1.0

    # Op_Server0 has Medium confidentiality (1.0), High availability (10.0)
    assert const.host_confidentiality[HOST_IDS['Op_Server0']] == 1.0
    assert const.host_availability[HOST_IDS['Op_Server0']] == 10.0


def test_service_vulnerability_mapping():
    """Test service-exploit vulnerability mapping."""
    const = create_scenario2_const()
    from jaxmarl.environments.cage.state import SERVICE_IDS, EXPLOIT_IDS

    # SSH is vulnerable to SSHBruteForce
    assert const.service_exploits[SERVICE_IDS['ssh'], EXPLOIT_IDS['SSHBruteForce']]

    # SSH is not vulnerable to EternalBlue
    assert not const.service_exploits[SERVICE_IDS['ssh'], EXPLOIT_IDS['EternalBlue']]

    # SMB is vulnerable to EternalBlue
    assert const.service_exploits[SERVICE_IDS['smb'], EXPLOIT_IDS['EternalBlue']]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
