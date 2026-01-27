"""CAGE-JAX action definitions and effects with configurable support."""

import jax
import jax.numpy as jnp
import chex
from functools import partial
from typing import Tuple

from jaxmarl.environments.cage.state import (
    CageState, CageConst,
    COMPROMISE_NONE, COMPROMISE_USER, COMPROMISE_PRIVILEGED,
    EXPLOIT_IDS, NUM_DECOY_TYPES, OS_LINUX, OS_WINDOWS, DECOY_IDS,
)
from jaxmarl.environments.cage.config import DECOY_OS_RESTRICTIONS, OS_ANY

# Build decoy OS restriction array from config (indexed by decoy type)
# Value: OS_ANY=-1, OS_LINUX=0, OS_WINDOWS=1
DECOY_OS_ARRAY = jnp.array([
    DECOY_OS_RESTRICTIONS.get(name, OS_ANY)
    for name, idx in sorted(DECOY_IDS.items(), key=lambda x: x[1])
], dtype=jnp.int32)

# Exploit-to-Decoy blocking mapping based on port/service matching
# Each exploit targets a specific service, and decoys for that service block it
# Decoy IDs: Apache=0, Femitter=1, HarakaSMPT=2, Smss=3, SSHD=4, Svchost=5, Tomcat=6, Vsftpd=7
# Exploit IDs: SSHBruteForce=0, FTPDirectoryTraversal=1, HTTPRFI=2, HTTPSRFI=3,
#              HarakaRCE=4, SQLInjection=5, EternalBlue=6, BlueKeep=7
EXPLOIT_BLOCKED_BY_DECOYS = jnp.array([
    # [exploit_idx, decoy_idx_1, decoy_idx_2] where -1 means no decoy
    [0, 4, -1],  # SSHBruteForce blocked by DecoySSHD (4)
    [1, 1, 7],   # FTPDirectoryTraversal blocked by DecoyFemitter (1) or DecoyVsftpd (7)
    [2, 0, 6],   # HTTPRFI blocked by DecoyApache (0) or DecoyTomcat (6)
    [3, 0, 6],   # HTTPSRFI blocked by DecoyApache (0) or DecoyTomcat (6)
    [4, 2, -1],  # HarakaRCE blocked by DecoyHarakaSMPT (2)
    [5, -1, -1], # SQLInjection - no decoy blocks it
    [6, 3, -1],  # EternalBlue blocked by DecoySmss (3)
    [7, 5, -1],  # BlueKeep blocked by DecoySvchost (5)
], dtype=jnp.int32)

# Decoy-to-Service conflict mapping: which services conflict with each decoy
# SERVICE_IDS: ssh=0, ftp=1, http=2, https=3, smtp=4, mysql=5, smb=6, rdp=7, tomcat=8, haraka=9
# Each decoy can only be deployed if its target port isn't already in use
# Format: [service_idx_1, service_idx_2] where -1 means no conflict
DECOY_SERVICE_CONFLICTS = jnp.array([
    [2, 8],   # DecoyApache (port 80) conflicts with http(2) or tomcat(8)
    [1, -1],  # DecoyFemitter (port 21) conflicts with ftp(1)
    [4, 9],   # DecoyHarakaSMPT (port 25) conflicts with smtp(4) or haraka(9)
    [6, -1],  # DecoySmss (port 139) conflicts with smb(6)
    [0, -1],  # DecoySSHD (port 22) conflicts with ssh(0)
    [7, -1],  # DecoySvchost (port 3389) conflicts with rdp(7)
    [2, 8],   # DecoyTomcat (port 80) conflicts with http(2) or tomcat(8)
    [1, -1],  # DecoyVsftpd (port 21) conflicts with ftp(1)
], dtype=jnp.int32)

# Default action space sizes for backward compatibility (Scenario 2 with 13 hosts)
NUM_HOSTS = 13
NUM_SUBNETS = 3
NUM_SERVICES = 10
NUM_EXPLOITS = 8

# Blue action encoding for default scenario (matching CybORG exactly)
# CybORG order: Sleep, Monitor, Analyse(13), Remove(13), Decoy(8×13), Restore(13)
BLUE_SLEEP = 0
BLUE_MONITOR = 1
BLUE_ANALYSE_START = 2
BLUE_REMOVE_START = BLUE_ANALYSE_START + NUM_HOSTS  # 15
BLUE_DECOY_START = BLUE_REMOVE_START + NUM_HOSTS  # 28

# All hosts can have decoys (matching CybORG)
DECOY_HOSTS = jnp.arange(0, NUM_HOSTS)
NUM_DECOY_HOSTS = NUM_HOSTS

# Restore comes AFTER decoys to match CybORG action order
BLUE_RESTORE_START = BLUE_DECOY_START + NUM_DECOY_HOSTS * NUM_DECOY_TYPES  # 132

# Total: 1 + 1 + 13 + 13 + (13*8) + 13 = 145 (matches CybORG)
NUM_BLUE_ACTIONS = 2 + NUM_HOSTS + NUM_HOSTS + NUM_DECOY_HOSTS * NUM_DECOY_TYPES + NUM_HOSTS

# Red action encoding for default scenario
RED_SLEEP = 0
RED_DISCOVER_SUBNET_START = 1
RED_SCAN_HOST_START = RED_DISCOVER_SUBNET_START + NUM_SUBNETS  # 4
RED_EXPLOIT_START = RED_SCAN_HOST_START + NUM_HOSTS  # 17
RED_PRIVESC_START = RED_EXPLOIT_START + NUM_EXPLOITS * NUM_HOSTS  # 121
RED_IMPACT_START = RED_PRIVESC_START + NUM_HOSTS  # 134

NUM_RED_ACTIONS = RED_IMPACT_START + NUM_HOSTS  # 147


def compute_blue_action_space_size(const: CageConst) -> int:
    """Compute total blue action space size for given configuration.

    CybORG order: sleep + monitor + analyse + remove + decoy + restore
    """
    # sleep(1) + monitor(1) + analyse(hosts) + remove(hosts) + decoy(hosts × types) + restore(hosts)
    return 2 + const.num_hosts + const.num_hosts + const.num_decoy_hosts * const.num_decoys + const.num_hosts


def compute_red_action_space_size(const: CageConst) -> int:
    """Compute total red action space size for given configuration."""
    # sleep + discover_subnet × subnets + scan × hosts + exploit × (exploits × hosts) + privesc × hosts + impact × hosts
    return (1 + const.num_subnets + const.num_hosts +
            const.num_exploits * const.num_hosts +
            const.num_hosts + const.num_hosts)


def get_blue_action_offsets(const: CageConst) -> Tuple[int, int, int, int]:
    """Get blue action offsets for given configuration.

    CybORG action order: Sleep, Monitor, Analyse, Remove, Decoy, Restore
    """
    analyse_start = 2
    remove_start = analyse_start + const.num_hosts
    decoy_start = remove_start + const.num_hosts
    restore_start = decoy_start + const.num_decoy_hosts * const.num_decoys
    return analyse_start, remove_start, decoy_start, restore_start


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
    CybORG action order: Sleep, Monitor, Analyse, Remove, Decoy, Restore
    """
    analyse_start, remove_start, decoy_start, restore_start = get_blue_action_offsets(const)

    action_type = jnp.where(
        action < analyse_start,
        action,  # Sleep=0, Monitor=1
        jnp.where(
            action < remove_start,
            2,  # Analyse
            jnp.where(
                action < decoy_start,
                3,  # Remove
                jnp.where(
                    action < restore_start,
                    5,  # Decoy
                    4,  # Restore
                )
            )
        )
    )

    # CybORG decoy action layout: decoy_type first, then host
    # action = decoy_start + decoy_type * num_decoy_hosts + host_index
    target_host = jnp.where(
        action < analyse_start,
        -1,  # Sleep/Monitor have no target
        jnp.where(
            action < remove_start,
            action - analyse_start,  # Analyse target
            jnp.where(
                action < decoy_start,
                action - remove_start,  # Remove target
                jnp.where(
                    action < restore_start,
                    const.decoy_host_indices[(action - decoy_start) % const.num_decoy_hosts],  # Decoy target
                    action - restore_start,  # Restore target
                )
            )
        )
    )

    decoy_type = jnp.where(
        (action >= decoy_start) & (action < restore_start),
        (action - decoy_start) // const.num_decoy_hosts,  # decoy_type = idx // num_hosts
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

    # Monitor (action_type == 1): detect activity on all hosts, clear unknown flags
    state = jax.lax.cond(
        action_type == 1,  # Monitor
        lambda s: _apply_monitor(s),
        lambda s: s,
        state,
    )

    # Analyse (action_type == 2): detect activity on target host, clear unknown flag
    state = jax.lax.cond(
        action_type == 2,  # Analyse
        lambda s: _apply_analyse(s, target_host),
        lambda s: s,
        state,
    )

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
        lambda s: _apply_decoy(s, target_host, decoy_type, const),
        lambda s: s,
        state,
    )

    return state


def _apply_monitor(state: CageState) -> CageState:
    """Monitor action: detect red activity on all hosts, clear unknown flags.

    CybORG behavior: Monitor only detects RECENT activity (actions taken this step),
    not pre-existing compromise state. The initial foothold is not detected unless
    Red takes a visible action.
    """
    return state.replace(
        host_activity_detected=state.host_activity_detected | state.red_activity_this_step,
        host_observation_unknown=jnp.zeros_like(state.host_observation_unknown),
    )


def _apply_analyse(state: CageState, target_host: int) -> CageState:
    """Analyse action: detect activity on target host, clear its unknown flag.

    CybORG behavior: Analyse detects recent activity on the target host.
    """
    has_activity = state.red_activity_this_step[target_host]
    new_detected = jnp.where(
        has_activity,
        state.host_activity_detected.at[target_host].set(True),
        state.host_activity_detected,
    )
    return state.replace(
        host_activity_detected=new_detected,
        host_observation_unknown=state.host_observation_unknown.at[target_host].set(False),
    )


def _apply_remove(state: CageState, target_host: int) -> CageState:
    """Remove action: kill user-level red sessions if activity was detected.

    CybORG behavior:
    - Only removes detected suspicious processes (requires prior Monitor/Analyse)
    - Cannot remove root/SYSTEM (privileged) processes
    - Sets observation to "Unknown" after Remove
    """
    was_detected = state.host_activity_detected[target_host]
    is_user_level = state.red_privilege[target_host] == COMPROMISE_USER

    # Remove only succeeds if activity was detected AND access is user-level (not privileged)
    can_remove = was_detected & is_user_level

    new_compromised = jnp.where(
        can_remove,
        COMPROMISE_NONE,
        state.host_compromised[target_host],
    )
    new_sessions = jnp.where(
        can_remove,
        0,
        state.red_sessions[target_host],
    )
    new_privilege = jnp.where(
        can_remove,
        COMPROMISE_NONE,
        state.red_privilege[target_host],
    )

    # Clear activity detected flag and set unknown flag after Remove
    new_activity_detected = state.host_activity_detected.at[target_host].set(False)
    new_observation_unknown = state.host_observation_unknown.at[target_host].set(True)

    return state.replace(
        host_compromised=state.host_compromised.at[target_host].set(new_compromised),
        red_sessions=state.red_sessions.at[target_host].set(new_sessions),
        red_privilege=state.red_privilege.at[target_host].set(new_privilege),
        host_activity_detected=new_activity_detected,
        host_observation_unknown=new_observation_unknown,
    )


def _apply_restore(state: CageState, target_host: int, const: CageConst) -> CageState:
    """Restore action: reset host to initial configuration.

    CybORG behavior: Restore clears Red sessions on most hosts, but preserves
    the initial foothold on User0 (host index 8). This matches CybORG tests
    where PrivEsc fails after Restore on exploited hosts, but succeeds on User0.
    """
    from jaxmarl.environments.cage.state import HOST_IDS

    is_initial_foothold = target_host == HOST_IDS['User0']

    new_sessions = jax.lax.cond(
        is_initial_foothold,
        lambda: state.red_sessions[target_host],
        lambda: jnp.array(0, dtype=state.red_sessions.dtype),
    )
    new_privilege = jax.lax.cond(
        is_initial_foothold,
        lambda: state.red_privilege[target_host],
        lambda: jnp.array(COMPROMISE_NONE, dtype=state.red_privilege.dtype),
    )

    return state.replace(
        host_compromised=state.host_compromised.at[target_host].set(COMPROMISE_NONE),
        host_services=state.host_services.at[target_host].set(const.initial_services[target_host]),
        host_decoys=state.host_decoys.at[target_host].set(jnp.zeros(const.num_decoys, dtype=jnp.bool_)),
        red_sessions=state.red_sessions.at[target_host].set(new_sessions),
        red_privilege=state.red_privilege.at[target_host].set(new_privilege),
        ot_service_stopped=state.ot_service_stopped.at[target_host].set(False),
        host_activity_detected=state.host_activity_detected.at[target_host].set(False),
        host_observation_unknown=state.host_observation_unknown.at[target_host].set(False),
    )


def _apply_decoy(state: CageState, target_host: int, decoy_type: int, const: CageConst) -> CageState:
    """Deploy decoy on target host if port is available (matching CybORG behavior).

    In CybORG, decoy deployment fails (returns success=FALSE) if the port is already
    in use by an existing service. We match this by only setting the decoy flag if
    the port is available.
    """
    # Check port conflict using DECOY_SERVICE_CONFLICTS
    conflict1 = DECOY_SERVICE_CONFLICTS[decoy_type, 0]
    conflict2 = DECOY_SERVICE_CONFLICTS[decoy_type, 1]
    has_conflict1 = jnp.where(conflict1 >= 0, state.host_services[target_host, conflict1], False)
    has_conflict2 = jnp.where(conflict2 >= 0, state.host_services[target_host, conflict2], False)
    port_available = ~has_conflict1 & ~has_conflict2

    # Only deploy if port is available
    new_decoy_value = jnp.where(port_available, True, state.host_decoys[target_host, decoy_type])

    return state.replace(
        host_decoys=state.host_decoys.at[target_host, decoy_type].set(new_decoy_value),
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
        lambda s: _apply_scan_host(s, target_host, const),
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
        lambda s: _apply_impact(s, target_host, const),
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

    new_discovered = jax.lax.fori_loop(0, num_hosts, update_discovered, state.red_discovered_hosts_jax)

    return state.replace(
        red_discovered_hosts_jax=new_discovered,
        last_red_action_success=can_reach,
    )


def _apply_scan_host(state: CageState, target_host: int, const: CageConst) -> CageState:
    """Scan target host for services.

    CybORG behavior: Scan works on any host that is routable from a session Red owns.
    Network routing is determined by subnet adjacency - Red needs a session in
    an adjacent subnet to scan the target.
    """
    num_hosts = const.num_hosts
    num_subnets = const.num_subnets
    target_subnet = const.host_subnet[target_host]

    # Find which subnets Red has sessions in
    red_subnets = jnp.zeros(num_subnets, dtype=jnp.bool_)

    def check_subnet(i, red_subnets):
        has_session = state.red_sessions[i] > 0
        subnet = const.host_subnet[i]
        return red_subnets.at[subnet].set(red_subnets[subnet] | has_session)

    red_subnets = jax.lax.fori_loop(0, num_hosts, check_subnet, red_subnets)

    # Can scan if Red has session in adjacent subnet (or same subnet)
    can_scan = jnp.any(red_subnets & const.subnet_adjacency[:, target_subnet])

    return state.replace(
        red_scanned_hosts_jax=state.red_scanned_hosts_jax.at[target_host].set(
            state.red_scanned_hosts_jax[target_host] | can_scan
        ),
        red_discovered_hosts_jax=state.red_discovered_hosts_jax.at[target_host].set(
            state.red_discovered_hosts_jax[target_host] | can_scan
        ),
        red_activity_this_step=state.red_activity_this_step.at[target_host].set(
            state.red_activity_this_step[target_host] | can_scan
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
    """Attempt to exploit target host. Deterministic success based on service + decoy.

    CybORG behavior: Different exploits give different privilege levels based on
    what user the vulnerable service runs as:
    - HarakaRCE (SMTP/Haraka) → root (service runs as root)
    - SSHBruteForce → pi/user (standard user bruteforce)
    - HTTPRFI/HTTPSRFI → NetworkService/www-data (limited web user)
    - Others → user-level

    Exploit indices that give PRIVILEGED: HarakaRCE (4)
    All others give USER level and require PrivEsc for root.
    """
    host_scanned = state.red_scanned_hosts_jax[target_host]

    services_on_host = state.host_services[target_host]
    exploit_vulnerabilities = const.service_exploits[:, exploit_type]
    has_vulnerable_service = jnp.any(services_on_host & exploit_vulnerabilities)

    # Check if any blocking decoy is deployed on the target host
    # EXPLOIT_BLOCKED_BY_DECOYS[exploit_type] gives [exploit_idx, decoy1, decoy2]
    blocking_decoys = EXPLOIT_BLOCKED_BY_DECOYS[exploit_type]
    decoy1 = blocking_decoys[1]
    decoy2 = blocking_decoys[2]
    decoy1_present = jnp.where(decoy1 >= 0, state.host_decoys[target_host, decoy1], False)
    decoy2_present = jnp.where(decoy2 >= 0, state.host_decoys[target_host, decoy2], False)
    decoy_present = decoy1_present | decoy2_present

    # Deterministic: success if host scanned, has vulnerable service, and no decoy
    success = host_scanned & has_vulnerable_service & ~decoy_present

    # HarakaRCE(4), EternalBlue(6), BlueKeep(7) give root directly (run as root/SYSTEM)
    gives_root = (exploit_type == 4) | (exploit_type == 6) | (exploit_type == 7)

    target_privilege = jnp.where(gives_root, COMPROMISE_PRIVILEGED, COMPROMISE_USER)

    new_compromised = jnp.where(
        success & (state.host_compromised[target_host] < target_privilege),
        target_privilege,
        state.host_compromised[target_host],
    )

    new_sessions = jnp.where(
        success,
        state.red_sessions[target_host] + 1,
        state.red_sessions[target_host],
    )

    new_privilege = jnp.where(
        success & (state.red_privilege[target_host] < target_privilege),
        target_privilege,
        state.red_privilege[target_host],
    )

    return state.replace(
        host_compromised=state.host_compromised.at[target_host].set(new_compromised),
        red_sessions=state.red_sessions.at[target_host].set(new_sessions),
        red_privilege=state.red_privilege.at[target_host].set(new_privilege),
        red_activity_this_step=state.red_activity_this_step.at[target_host].set(
            state.red_activity_this_step[target_host] | success
        ),
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
        red_activity_this_step=state.red_activity_this_step.at[target_host].set(
            state.red_activity_this_step[target_host] | success
        ),
        last_red_action_success=success,
    )


def _apply_impact(state: CageState, target_host: int, const: CageConst) -> CageState:
    """Impact action on operational host - stops OT service."""
    has_privileged = state.red_privilege[target_host] >= COMPROMISE_PRIVILEGED
    is_operational = const.operational_targets[target_host]
    success = has_privileged & is_operational

    new_ot_stopped = jnp.where(
        success,
        state.ot_service_stopped.at[target_host].set(True),
        state.ot_service_stopped,
    )

    return state.replace(
        ot_service_stopped=new_ot_stopped,
        red_activity_this_step=state.red_activity_this_step.at[target_host].set(
            state.red_activity_this_step[target_host] | success
        ),
        last_red_action_success=success,
    )


def get_blue_action_mask(state: CageState, const: CageConst) -> chex.Array:
    """Return valid action mask for blue agent."""
    action_size = compute_blue_action_space_size(const)
    mask = jnp.ones(action_size, dtype=jnp.bool_)
    analyse_start, remove_start, decoy_start, restore_start = get_blue_action_offsets(const)

    # Remove only valid if activity detected AND host has user-level (not privileged) access
    def check_remove(i, mask):
        activity_detected = state.host_activity_detected[i]
        is_user_level = state.red_privilege[i] == COMPROMISE_USER
        remove_valid = activity_detected & is_user_level
        return mask.at[remove_start + i].set(remove_valid)

    mask = jax.lax.fori_loop(0, const.num_hosts, check_remove, mask)

    # Decoys only if not already deployed and OS is compatible
    # (Port conflicts are checked at execution time, matching CybORG behavior)
    # CybORG action layout: decoy_type first, then host
    # action = decoy_start + decoy_type * num_decoy_hosts + host_index
    def check_decoy_type(d, mask):
        required_os = DECOY_OS_ARRAY[d]

        def check_host(i, mask):
            host_idx = const.decoy_host_indices[i]
            host_os = const.host_os[host_idx]
            decoy_action_idx = decoy_start + d * const.num_decoy_hosts + i
            already_deployed = state.host_decoys[host_idx, d]
            os_compatible = (required_os == OS_ANY) | (required_os == host_os)
            valid = ~already_deployed & os_compatible
            return mask.at[decoy_action_idx].set(valid)

        return jax.lax.fori_loop(0, const.num_decoy_hosts, check_host, mask)

    mask = jax.lax.fori_loop(0, const.num_decoys, check_decoy_type, mask)

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
        scan_valid = state.red_discovered_hosts_jax[i]
        return mask.at[scan_start + i].set(scan_valid)

    mask = jax.lax.fori_loop(0, const.num_hosts, check_scan, mask)

    # Exploit: need scanned host
    def check_exploit_type(e, mask):
        def check_exploit_host(i, mask):
            exploit_action_idx = exploit_start + e * const.num_hosts + i
            exploit_valid = state.red_scanned_hosts_jax[i]
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
