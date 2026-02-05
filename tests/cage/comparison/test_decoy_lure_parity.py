"""Parity tests for decoy lure behavior (decoys add services/ports)."""

import pytest
import jax
import jax.numpy as jnp

from tests.cage.differential.harness import DifferentialHarness, is_cyborg_available
from tests.cage.comparison.scenarios import (
    blue_decoy_host,
    red_discover_subnet,
    red_scan_host,
    red_exploit_host,
    DECOY_APACHE,
    DECOY_VSFTPD,
    SUBNET_USER,
    EXPLOIT_FTP,
)
from jaxmarl.environments.cage import CageEnv
from jaxmarl.environments.cage.state import HOST_IDS, SERVICE_IDS, EXPLOIT_IDS
from jaxmarl.environments.cage.actions import BLUE_SLEEP, RED_SLEEP, apply_blue_action
from jaxmarl.environments.cage.scripted_agents import _get_host_exploit_probabilistic
from tests.cage.differential.action_translator import get_exploit_class_from_action


requires_cyborg = pytest.mark.skipif(
    not is_cyborg_available(),
    reason="CybORG not installed"
)


def _get_red_ports(cyborg_env, hostname: str) -> list[int]:
    """Return list of ports Red has discovered for a hostname."""
    state = cyborg_env.environment_controller.state
    sessions = state.sessions.get('Red', {})
    if not sessions:
        return []
    # Use the first Red session (Scenario2 starts with one session)
    session = next(iter(sessions.values()))
    ip = cyborg_env.get_ip_map().get(hostname)
    if ip is None:
        return []
    ports = session.ports.get(ip, [])
    return list(ports)


@requires_cyborg
def test_decoy_apache_adds_http_service_and_port():
    """DecoyApache should add HTTP service in JAX and port 80 in CybORG."""
    harness = DifferentialHarness(seed=123, max_steps=3, verbose=False)
    harness.reset()

    host = "User2"
    host_idx = HOST_IDS[host]
    http_idx = SERVICE_IDS["http"]

    # Ensure host doesn't already have HTTP service
    assert not bool(harness.jax_env.const.initial_services[host_idx, http_idx])

    # Step 0: Blue deploys DecoyApache
    decoy_action = blue_decoy_host(host, DECOY_APACHE)
    harness.step(decoy_action, RED_SLEEP)

    # JAX: decoy should add HTTP service
    assert bool(harness.jax_state.host_services[host_idx, http_idx]), (
        "JAX should add HTTP service when DecoyApache is deployed"
    )

    # Step 1: Red discovers User subnet (needed before scan is valid)
    harness.step(BLUE_SLEEP, red_discover_subnet(SUBNET_USER))

    # Step 2: Red scans the host
    harness.step(BLUE_SLEEP, red_scan_host(host))

    # CybORG: Red should discover port 80 from the decoy
    ports = _get_red_ports(harness.cyborg_env, host)
    assert 80 in ports, f"CybORG should expose port 80 after DecoyApache, got ports={ports}"


def test_exploit_selection_changes_after_decoy_deploy():
    """Exploit selection should include the decoy service after deployment."""
    env = CageEnv(scenario='Scenario2')
    key = jax.random.PRNGKey(0)
    _obs, state = env.reset(key)
    const = env.const

    host = "User3"
    host_idx = HOST_IDS[host]
    ftp_service_idx = SERVICE_IDS["ftp"]
    ftp_exploit_idx = EXPLOIT_IDS["FTPDirectoryTraversal"]

    # User3 does not start with FTP service in Scenario2
    assert not bool(state.host_services[host_idx, ftp_service_idx])

    keys = jax.random.split(jax.random.PRNGKey(1), 200)
    picks_before = jax.vmap(
        lambda k: _get_host_exploit_probabilistic(host_idx, state.host_services, const, k)
    )(keys)
    assert int(jnp.sum(picks_before == ftp_exploit_idx)) == 0

    # Deploy DecoyVsftpd (adds FTP service)
    decoy_action = blue_decoy_host(host, DECOY_VSFTPD)
    state_after = apply_blue_action(state, jnp.array(decoy_action), const)
    assert bool(state_after.host_services[host_idx, ftp_service_idx])

    picks_after = jax.vmap(
        lambda k: _get_host_exploit_probabilistic(host_idx, state_after.host_services, const, k)
    )(keys)
    assert int(jnp.sum(picks_after == ftp_exploit_idx)) > 0


@requires_cyborg
def test_decoy_scan_exploit_sequence_parity():
    """Decoy -> scan -> exploit should target the decoy port in both envs."""
    harness = DifferentialHarness(seed=123, max_steps=5, verbose=False)
    harness.reset()

    host = "User3"
    host_idx = HOST_IDS[host]
    ftp_service_idx = SERVICE_IDS["ftp"]

    # Step 0: Blue deploys DecoyVsftpd on User3
    harness.step(blue_decoy_host(host, DECOY_VSFTPD), RED_SLEEP)
    assert bool(harness.jax_state.host_services[host_idx, ftp_service_idx])

    # Step 1: Red discovers User subnet
    harness.step(BLUE_SLEEP, red_discover_subnet(SUBNET_USER))
    # Step 2: Red scans the host
    harness.step(BLUE_SLEEP, red_scan_host(host))
    # Step 3: Red exploits using FTP (decoy port)
    result = harness.step(BLUE_SLEEP, red_exploit_host(host, EXPLOIT_FTP))

    cyborg_action = harness.cyborg_env.get_last_action('Red')
    exploit_class = get_exploit_class_from_action(cyborg_action)

    assert exploit_class == "FTPDirectoryTraversal", (
        f"CybORG should select FTP exploit after DecoyVsftpd, got {exploit_class}"
    )
    assert result.jax_action_success is False
