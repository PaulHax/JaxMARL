"""CAGE-JAX action definitions and effects."""

import jax
import jax.numpy as jnp
import chex
from functools import partial

from jaxmarl.environments.cage.state import (
    CageState, CageConst,
    NUM_HOSTS, NUM_SUBNETS, NUM_SERVICES, NUM_EXPLOITS, NUM_DECOY_TYPES,
    HOST_IDS, HOST_SUBNET, SERVICE_IDS, EXPLOIT_IDS, DECOY_IDS,
    COMPROMISE_NONE, COMPROMISE_USER, COMPROMISE_PRIVILEGED,
)

# Blue action encoding
# 0: Sleep
# 1: Monitor
# 2: Analyse
# 3-15: Remove (one per host)
# 16-28: Restore (one per host)
# 29-36: DecoyApache (one per Enterprise/Op host, 8 hosts: 5-12)
# 37-44: DecoySSHD (one per Enterprise/Op host)
# ... (simplified: just use decoy type × host)

BLUE_SLEEP = 0
BLUE_MONITOR = 1
BLUE_ANALYSE = 2
BLUE_REMOVE_START = 3
BLUE_RESTORE_START = 3 + NUM_HOSTS  # 16
BLUE_DECOY_START = BLUE_RESTORE_START + NUM_HOSTS  # 29

# For simplicity, decoys can be deployed on Enterprise/Operational hosts (indices 5-12 = 8 hosts)
DECOY_HOSTS = jnp.arange(5, 13)  # Enterprise0-Op_Server0
NUM_DECOY_HOSTS = 8

# Total blue actions: 3 + 13 (remove) + 13 (restore) + 8*8 (decoys) = 93
# Simplified: 3 + 13 + 13 + 8 = 37 (one decoy type)
NUM_BLUE_ACTIONS = 3 + NUM_HOSTS + NUM_HOSTS + NUM_DECOY_HOSTS * NUM_DECOY_TYPES  # 93

# Red action encoding
# 0: Sleep
# 1-3: DiscoverRemoteSystems per subnet
# 4-16: DiscoverNetworkServices per host (13 hosts)
# 17-120: Exploit (8 exploit types × 13 hosts = 104)
# 121-133: PrivilegeEscalate per host
# 134-146: Impact per host

RED_SLEEP = 0
RED_DISCOVER_SUBNET_START = 1
RED_SCAN_HOST_START = RED_DISCOVER_SUBNET_START + NUM_SUBNETS  # 4
RED_EXPLOIT_START = RED_SCAN_HOST_START + NUM_HOSTS  # 17
RED_PRIVESC_START = RED_EXPLOIT_START + NUM_EXPLOITS * NUM_HOSTS  # 121
RED_IMPACT_START = RED_PRIVESC_START + NUM_HOSTS  # 134

NUM_RED_ACTIONS = RED_IMPACT_START + NUM_HOSTS  # 147


def decode_blue_action(action: int) -> tuple:
    """Decode blue action into (action_type, target_host, decoy_type)."""
    action_type = jnp.where(
        action < BLUE_REMOVE_START,
        action,  # Sleep, Monitor, or Analyse
        jnp.where(
            action < BLUE_RESTORE_START,
            3,  # Remove
            jnp.where(
                action < BLUE_DECOY_START,
                4,  # Restore
                5,  # Decoy
            )
        )
    )

    target_host = jnp.where(
        action < BLUE_REMOVE_START,
        -1,  # No target for Sleep/Monitor/Analyse
        jnp.where(
            action < BLUE_RESTORE_START,
            action - BLUE_REMOVE_START,
            jnp.where(
                action < BLUE_DECOY_START,
                action - BLUE_RESTORE_START,
                DECOY_HOSTS[(action - BLUE_DECOY_START) // NUM_DECOY_TYPES],
            )
        )
    )

    decoy_type = jnp.where(
        action >= BLUE_DECOY_START,
        (action - BLUE_DECOY_START) % NUM_DECOY_TYPES,
        -1,
    )

    return action_type, target_host, decoy_type


def decode_red_action(action: int) -> tuple:
    """Decode red action into (action_type, target_host/subnet, exploit_type)."""
    action_type = jnp.where(
        action < RED_DISCOVER_SUBNET_START,
        0,  # Sleep
        jnp.where(
            action < RED_SCAN_HOST_START,
            1,  # DiscoverRemoteSystems
            jnp.where(
                action < RED_EXPLOIT_START,
                2,  # DiscoverNetworkServices
                jnp.where(
                    action < RED_PRIVESC_START,
                    3,  # Exploit
                    jnp.where(
                        action < RED_IMPACT_START,
                        4,  # PrivilegeEscalate
                        5,  # Impact
                    )
                )
            )
        )
    )

    # Target subnet for DiscoverRemoteSystems
    target_subnet = jnp.where(
        (action >= RED_DISCOVER_SUBNET_START) & (action < RED_SCAN_HOST_START),
        action - RED_DISCOVER_SUBNET_START,
        -1,
    )

    # Target host for scan/exploit/privesc/impact
    target_host = jnp.where(
        (action >= RED_SCAN_HOST_START) & (action < RED_EXPLOIT_START),
        action - RED_SCAN_HOST_START,
        jnp.where(
            (action >= RED_EXPLOIT_START) & (action < RED_PRIVESC_START),
            (action - RED_EXPLOIT_START) % NUM_HOSTS,
            jnp.where(
                (action >= RED_PRIVESC_START) & (action < RED_IMPACT_START),
                action - RED_PRIVESC_START,
                jnp.where(
                    action >= RED_IMPACT_START,
                    action - RED_IMPACT_START,
                    -1,
                )
            )
        )
    )

    exploit_type = jnp.where(
        (action >= RED_EXPLOIT_START) & (action < RED_PRIVESC_START),
        (action - RED_EXPLOIT_START) // NUM_HOSTS,
        -1,
    )

    return action_type, target_subnet, target_host, exploit_type


@partial(jax.jit, static_argnums=[])
def apply_blue_action(state: CageState, action: chex.Array, const: CageConst) -> CageState:
    """Apply blue agent action to state."""
    action_type, target_host, decoy_type = decode_blue_action(action)

    # Remove action: clear compromise and red sessions on target host
    state = jax.lax.cond(
        action_type == 3,  # Remove
        lambda s: _apply_remove(s, target_host),
        lambda s: s,
        state,
    )

    # Restore action: reset host to initial state
    state = jax.lax.cond(
        action_type == 4,  # Restore
        lambda s: _apply_restore(s, target_host, const),
        lambda s: s,
        state,
    )

    # Decoy action: deploy decoy on target host
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
        host_decoys=state.host_decoys.at[target_host].set(jnp.zeros(NUM_DECOY_TYPES, dtype=jnp.bool_)),
        red_sessions=state.red_sessions.at[target_host].set(0),
        red_privilege=state.red_privilege.at[target_host].set(COMPROMISE_NONE),
    )


def _apply_decoy(state: CageState, target_host: int, decoy_type: int) -> CageState:
    """Deploy decoy on target host."""
    return state.replace(
        host_decoys=state.host_decoys.at[target_host, decoy_type].set(True),
    )


@partial(jax.jit, static_argnums=[])
def apply_red_action(
    state: CageState,
    action: chex.Array,
    const: CageConst,
    key: chex.PRNGKey
) -> CageState:
    """Apply red agent action to state."""
    action_type, target_subnet, target_host, exploit_type = decode_red_action(action)

    # DiscoverRemoteSystems: discover hosts in target subnet
    state = jax.lax.cond(
        action_type == 1,  # DiscoverRemoteSystems
        lambda s: _apply_discover_subnet(s, target_subnet, const),
        lambda s: s,
        state,
    )

    # DiscoverNetworkServices: scan target host
    state = jax.lax.cond(
        action_type == 2,  # DiscoverNetworkServices
        lambda s: _apply_scan_host(s, target_host),
        lambda s: s,
        state,
    )

    # Exploit: attempt to exploit target host
    key, subkey = jax.random.split(key)
    state = jax.lax.cond(
        action_type == 3,  # Exploit
        lambda s: _apply_exploit(s, target_host, exploit_type, const, subkey),
        lambda s: s,
        state,
    )

    # PrivilegeEscalate: escalate privileges on target host
    key, subkey = jax.random.split(key)
    state = jax.lax.cond(
        action_type == 4,  # PrivilegeEscalate
        lambda s: _apply_privesc(s, target_host, subkey),
        lambda s: s,
        state,
    )

    # Impact: disrupts services (for Op_Server0)
    state = jax.lax.cond(
        action_type == 5,  # Impact
        lambda s: _apply_impact(s, target_host),
        lambda s: s,
        state,
    )

    return state


def _apply_discover_subnet(state: CageState, target_subnet: int, const: CageConst) -> CageState:
    """Discover all hosts in target subnet if Red has access to adjacent subnet."""
    # Check if Red has a session in a subnet that can reach target_subnet
    red_subnets = jnp.zeros(NUM_SUBNETS, dtype=jnp.bool_)
    for i in range(NUM_HOSTS):
        has_session = state.red_sessions[i] > 0
        subnet = HOST_SUBNET[i]
        red_subnets = red_subnets.at[subnet].set(red_subnets[subnet] | has_session)

    can_reach = jnp.any(red_subnets & const.subnet_adjacency[:, target_subnet])

    # Discover all hosts in target subnet
    new_discovered = state.red_discovered_hosts.copy()
    for i in range(NUM_HOSTS):
        is_in_subnet = HOST_SUBNET[i] == target_subnet
        new_discovered = new_discovered.at[i].set(
            new_discovered[i] | (can_reach & is_in_subnet)
        )

    success = can_reach

    return state.replace(
        red_discovered_hosts=new_discovered,
        last_red_action_success=success,
    )


def _apply_scan_host(state: CageState, target_host: int) -> CageState:
    """Scan target host for services."""
    # Can only scan if host is discovered
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
    # Preconditions:
    # 1. Host must be scanned
    # 2. Host must have vulnerable service running
    # 3. No decoy matching the exploit type

    host_scanned = state.red_scanned_hosts[target_host]

    # Check if any service on host is vulnerable to this exploit
    services_on_host = state.host_services[target_host]  # (NUM_SERVICES,)
    exploit_vulnerabilities = const.service_exploits[:, exploit_type]  # (NUM_SERVICES,)
    has_vulnerable_service = jnp.any(services_on_host & exploit_vulnerabilities)

    # Check for decoy (simplified: decoy blocks if it matches exploit's target service)
    # Map exploit to decoy type (simplified: exploit 0 -> decoy 0, etc.)
    decoy_present = state.host_decoys[target_host, exploit_type % NUM_DECOY_TYPES]

    # Probabilistic success (0.8 for SSH brute force, 0.9 for others)
    success_prob = jnp.where(exploit_type == EXPLOIT_IDS['SSHBruteForce'], 0.8, 0.9)
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
    # Precondition: must have user-level session
    has_user_session = state.red_privilege[target_host] >= COMPROMISE_USER

    # Probabilistic success (0.9)
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
    """Impact action on operational host (disrupts services)."""
    # Precondition: must have privileged access
    has_privileged = state.red_privilege[target_host] >= COMPROMISE_PRIVILEGED

    # For now, impact is a successful action if we have privileged access
    success = has_privileged

    return state.replace(
        last_red_action_success=success,
    )


def get_blue_action_mask(state: CageState, const: CageConst) -> chex.Array:
    """Return valid action mask for blue agent."""
    mask = jnp.ones(NUM_BLUE_ACTIONS, dtype=jnp.bool_)

    # Sleep, Monitor, Analyse always valid
    # Remove only valid if host is compromised
    for i in range(NUM_HOSTS):
        remove_valid = state.host_compromised[i] > 0
        mask = mask.at[BLUE_REMOVE_START + i].set(remove_valid)

    # Restore always valid (but costly)

    # Decoys only if not already deployed on that host
    for i, host_idx in enumerate(DECOY_HOSTS):
        for d in range(NUM_DECOY_TYPES):
            decoy_action_idx = BLUE_DECOY_START + i * NUM_DECOY_TYPES + d
            already_deployed = state.host_decoys[host_idx, d]
            mask = mask.at[decoy_action_idx].set(~already_deployed)

    return mask


def get_red_action_mask(state: CageState, const: CageConst) -> chex.Array:
    """Return valid action mask for red agent."""
    mask = jnp.ones(NUM_RED_ACTIONS, dtype=jnp.bool_)

    # Sleep always valid

    # DiscoverRemoteSystems: need session in connected subnet
    for subnet in range(NUM_SUBNETS):
        # Check if Red has session in any subnet that can reach this one
        has_access = False
        for i in range(NUM_HOSTS):
            host_subnet = HOST_SUBNET[i]
            has_session = state.red_sessions[i] > 0
            can_reach = const.subnet_adjacency[host_subnet, subnet]
            has_access = has_access | (has_session & can_reach)
        mask = mask.at[RED_DISCOVER_SUBNET_START + subnet].set(has_access)

    # DiscoverNetworkServices: need discovered host
    for i in range(NUM_HOSTS):
        scan_valid = state.red_discovered_hosts[i]
        mask = mask.at[RED_SCAN_HOST_START + i].set(scan_valid)

    # Exploit: need scanned host
    for e in range(NUM_EXPLOITS):
        for i in range(NUM_HOSTS):
            exploit_action_idx = RED_EXPLOIT_START + e * NUM_HOSTS + i
            exploit_valid = state.red_scanned_hosts[i]
            mask = mask.at[exploit_action_idx].set(exploit_valid)

    # PrivilegeEscalate: need user session
    for i in range(NUM_HOSTS):
        privesc_valid = state.red_privilege[i] >= COMPROMISE_USER
        mask = mask.at[RED_PRIVESC_START + i].set(privesc_valid)

    # Impact: need privileged session
    for i in range(NUM_HOSTS):
        impact_valid = state.red_privilege[i] >= COMPROMISE_PRIVILEGED
        mask = mask.at[RED_IMPACT_START + i].set(impact_valid)

    return mask
