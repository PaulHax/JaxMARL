"""CAGE-JAX action definitions and effects with configurable support."""

import jax
import jax.numpy as jnp
import chex
from functools import partial
from typing import Tuple

from jaxmarl.environments.cage.state import (
    CageState, CageConst,
    COMPROMISE_NONE, COMPROMISE_USER, COMPROMISE_PRIVILEGED,
    EXPLOIT_IDS, NUM_DECOY_TYPES,
)

# Default action space sizes for backward compatibility (Scenario 2 with 13 hosts)
NUM_HOSTS = 13
NUM_SUBNETS = 3
NUM_SERVICES = 10
NUM_EXPLOITS = 8

# Blue action encoding for default scenario (matching CybORG exactly)
BLUE_SLEEP = 0
BLUE_MONITOR = 1
BLUE_ANALYSE_START = 2
BLUE_REMOVE_START = BLUE_ANALYSE_START + NUM_HOSTS  # 15
BLUE_RESTORE_START = BLUE_REMOVE_START + NUM_HOSTS  # 28
BLUE_DECOY_START = BLUE_RESTORE_START + NUM_HOSTS  # 41

# All hosts can have decoys (matching CybORG)
DECOY_HOSTS = jnp.arange(0, NUM_HOSTS)
NUM_DECOY_HOSTS = NUM_HOSTS

# Total: 1 + 1 + 13 + 13 + 13 + 13*8 = 145 (matches CybORG)
NUM_BLUE_ACTIONS = 2 + NUM_HOSTS + NUM_HOSTS + NUM_HOSTS + NUM_DECOY_HOSTS * NUM_DECOY_TYPES

# Red action encoding for default scenario
RED_SLEEP = 0
RED_DISCOVER_SUBNET_START = 1
RED_SCAN_HOST_START = RED_DISCOVER_SUBNET_START + NUM_SUBNETS  # 4
RED_EXPLOIT_START = RED_SCAN_HOST_START + NUM_HOSTS  # 17
RED_PRIVESC_START = RED_EXPLOIT_START + NUM_EXPLOITS * NUM_HOSTS  # 121
RED_IMPACT_START = RED_PRIVESC_START + NUM_HOSTS  # 134

NUM_RED_ACTIONS = RED_IMPACT_START + NUM_HOSTS  # 147


def compute_blue_action_space_size(const: CageConst) -> int:
    """Compute total blue action space size for given configuration."""
    # sleep + monitor + analyse per host + remove per host + restore per host + decoy per (host × decoy_type)
    return 2 + const.num_hosts + const.num_hosts + const.num_hosts + const.num_decoy_hosts * const.num_decoys


def compute_red_action_space_size(const: CageConst) -> int:
    """Compute total red action space size for given configuration."""
    # sleep + discover_subnet × subnets + scan × hosts + exploit × (exploits × hosts) + privesc × hosts + impact × hosts
    return (1 + const.num_subnets + const.num_hosts +
            const.num_exploits * const.num_hosts +
            const.num_hosts + const.num_hosts)


def get_blue_action_offsets(const: CageConst) -> Tuple[int, int, int, int]:
    """Get blue action offsets for given configuration."""
    analyse_start = 2
    remove_start = analyse_start + const.num_hosts
    restore_start = remove_start + const.num_hosts
    decoy_start = restore_start + const.num_hosts
    return analyse_start, remove_start, restore_start, decoy_start


def get_red_action_offsets(const: CageConst) -> Tuple[int, int, int, int, int]:
    """Get red action offsets for given configuration."""
    discover_start = 1
    scan_start = discover_start + const.num_subnets
    exploit_start = scan_start + const.num_hosts
    privesc_start = exploit_start + const.num_exploits * const.num_hosts
    impact_start = privesc_start + const.num_hosts
    return discover_start, scan_start, exploit_start, privesc_start, impact_start


def decode_blue_action(action: int, const: CageConst) -> Tuple[chex.Array, chex.Array, chex.Array]:
    """Decode blue action into (action_type, target_host, decoy_type).

    Action types: 0=Sleep, 1=Monitor, 2=Analyse, 3=Remove, 4=Restore, 5=Decoy
    """
    analyse_start, remove_start, restore_start, decoy_start = get_blue_action_offsets(const)

    action_type = jnp.where(
        action < analyse_start,
        action,  # Sleep=0, Monitor=1
        jnp.where(
            action < remove_start,
            2,  # Analyse
            jnp.where(
                action < restore_start,
                3,  # Remove
                jnp.where(
                    action < decoy_start,
                    4,  # Restore
                    5,  # Decoy
                )
            )
        )
    )

    target_host = jnp.where(
        action < analyse_start,
        -1,  # Sleep/Monitor have no target
        jnp.where(
            action < remove_start,
            action - analyse_start,  # Analyse target
            jnp.where(
                action < restore_start,
                action - remove_start,  # Remove target
                jnp.where(
                    action < decoy_start,
                    action - restore_start,  # Restore target
                    const.decoy_host_indices[(action - decoy_start) // const.num_decoys],  # Decoy target
                )
            )
        )
    )

    decoy_type = jnp.where(
        action >= decoy_start,
        (action - decoy_start) % const.num_decoys,
        -1,
    )

    return action_type, target_host, decoy_type


def decode_red_action(action: int, const: CageConst) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
    """Decode red action into (action_type, target_subnet, target_host, exploit_type)."""
    discover_start, scan_start, exploit_start, privesc_start, impact_start = get_red_action_offsets(const)

    action_type = jnp.where(
        action < discover_start,
        0,  # Sleep
        jnp.where(
            action < scan_start,
            1,  # DiscoverRemoteSystems
            jnp.where(
                action < exploit_start,
                2,  # DiscoverNetworkServices
                jnp.where(
                    action < privesc_start,
                    3,  # Exploit
                    jnp.where(
                        action < impact_start,
                        4,  # PrivilegeEscalate
                        5,  # Impact
                    )
                )
            )
        )
    )

    target_subnet = jnp.where(
        (action >= discover_start) & (action < scan_start),
        action - discover_start,
        -1,
    )

    target_host = jnp.where(
        (action >= scan_start) & (action < exploit_start),
        action - scan_start,
        jnp.where(
            (action >= exploit_start) & (action < privesc_start),
            (action - exploit_start) % const.num_hosts,
            jnp.where(
                (action >= privesc_start) & (action < impact_start),
                action - privesc_start,
                jnp.where(
                    action >= impact_start,
                    action - impact_start,
                    -1,
                )
            )
        )
    )

    exploit_type = jnp.where(
        (action >= exploit_start) & (action < privesc_start),
        (action - exploit_start) // const.num_hosts,
        -1,
    )

    return action_type, target_subnet, target_host, exploit_type


def apply_blue_action(state: CageState, action: chex.Array, const: CageConst) -> CageState:
    """Apply blue agent action to state."""
    action_type, target_host, decoy_type = decode_blue_action(action, const)

    state = jax.lax.cond(
        action_type == 3,  # Remove
        lambda s: _apply_remove(s, target_host),
        lambda s: s,
        state,
    )

    state = jax.lax.cond(
        action_type == 4,  # Restore
        lambda s: _apply_restore(s, target_host, const),
        lambda s: s,
        state,
    )

    state = jax.lax.cond(
        action_type == 5,  # Decoy
        lambda s: _apply_decoy(s, target_host, decoy_type),
        lambda s: s,
        state,
    )

    return state


def _apply_remove(state: CageState, target_host: int) -> CageState:
    """Remove action: kill red sessions and clear user-level compromise."""
    return state.replace(
        host_compromised=state.host_compromised.at[target_host].set(COMPROMISE_NONE),
        red_sessions=state.red_sessions.at[target_host].set(0),
        red_privilege=state.red_privilege.at[target_host].set(COMPROMISE_NONE),
    )


def _apply_restore(state: CageState, target_host: int, const: CageConst) -> CageState:
    """Restore action: reset host to initial configuration."""
    return state.replace(
        host_compromised=state.host_compromised.at[target_host].set(COMPROMISE_NONE),
        host_services=state.host_services.at[target_host].set(const.initial_services[target_host]),
        host_decoys=state.host_decoys.at[target_host].set(jnp.zeros(const.num_decoys, dtype=jnp.bool_)),
        red_sessions=state.red_sessions.at[target_host].set(0),
        red_privilege=state.red_privilege.at[target_host].set(COMPROMISE_NONE),
    )


def _apply_decoy(state: CageState, target_host: int, decoy_type: int) -> CageState:
    """Deploy decoy on target host."""
    return state.replace(
        host_decoys=state.host_decoys.at[target_host, decoy_type].set(True),
    )


def apply_red_action(
    state: CageState,
    action: chex.Array,
    const: CageConst,
    key: chex.PRNGKey
) -> CageState:
    """Apply red agent action to state."""
    action_type, target_subnet, target_host, exploit_type = decode_red_action(action, const)

    state = jax.lax.cond(
        action_type == 1,  # DiscoverRemoteSystems
        lambda s: _apply_discover_subnet(s, target_subnet, const),
        lambda s: s,
        state,
    )

    state = jax.lax.cond(
        action_type == 2,  # DiscoverNetworkServices
        lambda s: _apply_scan_host(s, target_host),
        lambda s: s,
        state,
    )

    key, subkey = jax.random.split(key)
    state = jax.lax.cond(
        action_type == 3,  # Exploit
        lambda s: _apply_exploit(s, target_host, exploit_type, const, subkey),
        lambda s: s,
        state,
    )

    key, subkey = jax.random.split(key)
    state = jax.lax.cond(
        action_type == 4,  # PrivilegeEscalate
        lambda s: _apply_privesc(s, target_host, subkey),
        lambda s: s,
        state,
    )

    state = jax.lax.cond(
        action_type == 5,  # Impact
        lambda s: _apply_impact(s, target_host),
        lambda s: s,
        state,
    )

    return state


def _apply_discover_subnet(state: CageState, target_subnet: int, const: CageConst) -> CageState:
    """Discover all hosts in target subnet if Red has access to adjacent subnet."""
    num_hosts = const.num_hosts
    num_subnets = const.num_subnets

    red_subnets = jnp.zeros(num_subnets, dtype=jnp.bool_)

    def check_subnet(i, red_subnets):
        has_session = state.red_sessions[i] > 0
        subnet = const.host_subnet[i]
        return red_subnets.at[subnet].set(red_subnets[subnet] | has_session)

    red_subnets = jax.lax.fori_loop(0, num_hosts, check_subnet, red_subnets)

    can_reach = jnp.any(red_subnets & const.subnet_adjacency[:, target_subnet])

    def update_discovered(i, new_discovered):
        is_in_subnet = const.host_subnet[i] == target_subnet
        return new_discovered.at[i].set(new_discovered[i] | (can_reach & is_in_subnet))

    new_discovered = jax.lax.fori_loop(0, num_hosts, update_discovered, state.red_discovered_hosts)

    return state.replace(
        red_discovered_hosts=new_discovered,
        last_red_action_success=can_reach,
    )


def _apply_scan_host(state: CageState, target_host: int) -> CageState:
    """Scan target host for services."""
    can_scan = state.red_discovered_hosts[target_host]

    return state.replace(
        red_scanned_hosts=state.red_scanned_hosts.at[target_host].set(
            state.red_scanned_hosts[target_host] | can_scan
        ),
        last_red_action_success=can_scan,
    )


def _apply_exploit(
    state: CageState,
    target_host: int,
    exploit_type: int,
    const: CageConst,
    key: chex.PRNGKey
) -> CageState:
    """Attempt to exploit target host."""
    host_scanned = state.red_scanned_hosts[target_host]

    services_on_host = state.host_services[target_host]
    exploit_vulnerabilities = const.service_exploits[:, exploit_type]
    has_vulnerable_service = jnp.any(services_on_host & exploit_vulnerabilities)

    decoy_present = state.host_decoys[target_host, exploit_type % const.num_decoys]

    # SSH brute force (exploit 0) has 0.8 success rate, others 0.9
    success_prob = jnp.where(exploit_type == 0, 0.8, 0.9)
    random_success = jax.random.uniform(key) < success_prob

    can_exploit = host_scanned & has_vulnerable_service & ~decoy_present
    success = can_exploit & random_success

    new_compromised = jnp.where(
        success & (state.host_compromised[target_host] < COMPROMISE_USER),
        COMPROMISE_USER,
        state.host_compromised[target_host],
    )

    new_sessions = jnp.where(
        success,
        state.red_sessions[target_host] + 1,
        state.red_sessions[target_host],
    )

    new_privilege = jnp.where(
        success & (state.red_privilege[target_host] < COMPROMISE_USER),
        COMPROMISE_USER,
        state.red_privilege[target_host],
    )

    return state.replace(
        host_compromised=state.host_compromised.at[target_host].set(new_compromised),
        red_sessions=state.red_sessions.at[target_host].set(new_sessions),
        red_privilege=state.red_privilege.at[target_host].set(new_privilege),
        last_red_action_success=success,
    )


def _apply_privesc(state: CageState, target_host: int, key: chex.PRNGKey) -> CageState:
    """Escalate privileges on target host."""
    has_user_session = state.red_privilege[target_host] >= COMPROMISE_USER
    random_success = jax.random.uniform(key) < 0.9

    success = has_user_session & random_success

    new_privilege = jnp.where(
        success,
        COMPROMISE_PRIVILEGED,
        state.red_privilege[target_host],
    )

    new_compromised = jnp.where(
        success,
        COMPROMISE_PRIVILEGED,
        state.host_compromised[target_host],
    )

    return state.replace(
        red_privilege=state.red_privilege.at[target_host].set(new_privilege),
        host_compromised=state.host_compromised.at[target_host].set(new_compromised),
        last_red_action_success=success,
    )


def _apply_impact(state: CageState, target_host: int) -> CageState:
    """Impact action on operational host."""
    has_privileged = state.red_privilege[target_host] >= COMPROMISE_PRIVILEGED
    success = has_privileged

    return state.replace(
        last_red_action_success=success,
    )


def get_blue_action_mask(state: CageState, const: CageConst) -> chex.Array:
    """Return valid action mask for blue agent."""
    action_size = compute_blue_action_space_size(const)
    mask = jnp.ones(action_size, dtype=jnp.bool_)
    analyse_start, remove_start, restore_start, decoy_start = get_blue_action_offsets(const)

    # Remove only valid if host is compromised
    def check_remove(i, mask):
        remove_valid = state.host_compromised[i] > 0
        return mask.at[remove_start + i].set(remove_valid)

    mask = jax.lax.fori_loop(0, const.num_hosts, check_remove, mask)

    # Decoys only if not already deployed
    def check_decoy_host(i, mask):
        host_idx = const.decoy_host_indices[i]

        def check_decoy_type(d, mask):
            decoy_action_idx = decoy_start + i * const.num_decoys + d
            already_deployed = state.host_decoys[host_idx, d]
            return mask.at[decoy_action_idx].set(~already_deployed)

        return jax.lax.fori_loop(0, const.num_decoys, check_decoy_type, mask)

    mask = jax.lax.fori_loop(0, const.num_decoy_hosts, check_decoy_host, mask)

    return mask


def get_red_action_mask(state: CageState, const: CageConst) -> chex.Array:
    """Return valid action mask for red agent."""
    action_size = compute_red_action_space_size(const)
    mask = jnp.ones(action_size, dtype=jnp.bool_)
    discover_start, scan_start, exploit_start, privesc_start, impact_start = get_red_action_offsets(const)

    # DiscoverRemoteSystems: need session in connected subnet
    def check_discover(subnet, mask):
        def check_host_access(i, has_access):
            host_subnet = const.host_subnet[i]
            has_session = state.red_sessions[i] > 0
            can_reach = const.subnet_adjacency[host_subnet, subnet]
            return has_access | (has_session & can_reach)

        has_access = jax.lax.fori_loop(0, const.num_hosts, check_host_access, False)
        return mask.at[discover_start + subnet].set(has_access)

    mask = jax.lax.fori_loop(0, const.num_subnets, check_discover, mask)

    # DiscoverNetworkServices: need discovered host
    def check_scan(i, mask):
        scan_valid = state.red_discovered_hosts[i]
        return mask.at[scan_start + i].set(scan_valid)

    mask = jax.lax.fori_loop(0, const.num_hosts, check_scan, mask)

    # Exploit: need scanned host
    def check_exploit_type(e, mask):
        def check_exploit_host(i, mask):
            exploit_action_idx = exploit_start + e * const.num_hosts + i
            exploit_valid = state.red_scanned_hosts[i]
            return mask.at[exploit_action_idx].set(exploit_valid)

        return jax.lax.fori_loop(0, const.num_hosts, check_exploit_host, mask)

    mask = jax.lax.fori_loop(0, const.num_exploits, check_exploit_type, mask)

    # PrivilegeEscalate: need user session
    def check_privesc(i, mask):
        privesc_valid = state.red_privilege[i] >= COMPROMISE_USER
        return mask.at[privesc_start + i].set(privesc_valid)

    mask = jax.lax.fori_loop(0, const.num_hosts, check_privesc, mask)

    # Impact: need privileged session
    def check_impact(i, mask):
        impact_valid = state.red_privilege[i] >= COMPROMISE_PRIVILEGED
        return mask.at[impact_start + i].set(impact_valid)

    mask = jax.lax.fori_loop(0, const.num_hosts, check_impact, mask)

    return mask
