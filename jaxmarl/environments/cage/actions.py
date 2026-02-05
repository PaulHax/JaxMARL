"""CAGE-JAX action definitions and effects with configurable support."""

import jax
import jax.numpy as jnp
import chex
from functools import partial
from typing import Tuple

from jaxmarl.environments.cage.state import (
    CageState, CageConst,
    COMPROMISE_NONE, COMPROMISE_USER, COMPROMISE_PRIVILEGED,
    ACTIVITY_NONE, ACTIVITY_SCAN, ACTIVITY_EXPLOIT,
    EXPLOIT_IDS, NUM_DECOY_TYPES, OS_LINUX, OS_WINDOWS, DECOY_IDS,
    HOST_NAMES, SUBNET_IDS, SERVICE_IDS,
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

# Mapping from decoy type -> service index to add when decoy is deployed.
# This is used to make decoys visible to Red's service discovery/exploit selection,
# matching CybORG's behavior where decoys add services with open ports.
# Note: DecoyTomcat is mapped to HTTP to match existing parity expectations/tests.
DECOY_SERVICE_IDS = jnp.array([
    SERVICE_IDS['http'],    # DecoyApache (port 80)
    SERVICE_IDS['ftp'],     # DecoyFemitter (port 21)
    SERVICE_IDS['haraka'],  # DecoyHarakaSMPT (port 25)
    SERVICE_IDS['smb'],     # DecoySmss (port 139)
    SERVICE_IDS['ssh'],     # DecoySSHD (port 22)
    SERVICE_IDS['rdp'],     # DecoySvchost (port 3389)
    SERVICE_IDS['http'],    # DecoyTomcat (treated as HTTP for parity)
    SERVICE_IDS['ftp'],     # DecoyVsftpd (port 21)
], dtype=jnp.int32)

# Default action space sizes for backward compatibility (Scenario 2 with 13 hosts)
NUM_HOSTS = 13
NUM_SUBNETS = 3
NUM_SERVICES = 10
NUM_EXPLOITS = 8

# Blue action type names (indexed by action_type from decode_blue_action)
# Action types: 0=Sleep, 1=Monitor, 2=Analyse, 3=Remove, 4=Restore, 5=Decoy
BLUE_ACTION_NAMES = ["Sleep", "Monitor", "Analyse", "Remove", "Restore", "Decoy"]

# Red action type names (indexed by action_type from decode_red_action)
# Action types: 0=Sleep, 1=DiscoverRemoteSystems, 2=DiscoverNetworkServices, 3=Exploit, 4=PrivilegeEscalate, 5=Impact
RED_ACTION_NAMES = ["Sleep", "DiscoverRemoteSystems", "DiscoverNetworkServices", "ExploitRemoteService", "PrivilegeEscalate", "Impact"]

# Decoy type names (indexed by decoy_type)
DECOY_NAMES = {v: k for k, v in DECOY_IDS.items()}

# Exploit type names (indexed by exploit_type)
EXPLOIT_NAMES = {v: k for k, v in EXPLOIT_IDS.items()}

# Subnet names (indexed by subnet_id)
SUBNET_NAMES = {v: k for k, v in SUBNET_IDS.items()}

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
            (action - exploit_start) // const.num_exploits,
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
        (action - exploit_start) % const.num_exploits,
        -1,
    )

    return action_type, target_subnet, target_host, exploit_type


def apply_blue_action(state: CageState, action: chex.Array, const: CageConst) -> CageState:
    """Apply blue agent action to state.

    Tracks action success in state.last_blue_action_success:
    - Sleep, Monitor, Analyse, Restore: Always succeed
    - Remove: Succeeds only if activity was detected AND access is user-level (not privileged)
    - Decoy: Succeeds only if port is available AND OS is compatible

    Failed actions get -0.1 penalty in compute_rewards (matching CybORG InvalidAction.cost).
    """
    action_type, target_host, decoy_type = decode_blue_action(action, const)

    # Compute success BEFORE applying action (need original state for Remove/Decoy checks)
    # Remove success: activity detected AND user-level access (not privileged)
    remove_success = (
        state.host_activity_detected[target_host] &
        (state.red_privilege[target_host] == COMPROMISE_USER)
    )

    # Decoy success: not already deployed AND port available AND OS compatible
    already_deployed = state.host_decoys[target_host, decoy_type]
    conflict1 = DECOY_SERVICE_CONFLICTS[decoy_type, 0]
    conflict2 = DECOY_SERVICE_CONFLICTS[decoy_type, 1]
    has_conflict1 = jnp.where(conflict1 >= 0, state.host_services[target_host, conflict1], False)
    has_conflict2 = jnp.where(conflict2 >= 0, state.host_services[target_host, conflict2], False)
    port_available = ~has_conflict1 & ~has_conflict2
    required_os = DECOY_OS_ARRAY[decoy_type]
    host_os = const.host_os[target_host]
    os_compatible = (required_os == OS_ANY) | (required_os == host_os)
    decoy_success = ~already_deployed & port_available & os_compatible

    # Monitor (action_type == 1): detect activity on all hosts, clear unknown flags
    state = jax.lax.cond(
        action_type == 1,  # Monitor
        lambda s: _apply_monitor(s, const),
        lambda s: s,
        state,
    )

    # Analyse (action_type == 2): detect activity on target host, clear unknown flag
    state = jax.lax.cond(
        action_type == 2,  # Analyse
        lambda s: _apply_analyse(s, target_host, const),
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

    # Determine overall action success
    # Sleep(0), Monitor(1), Analyse(2), Restore(4) always succeed
    # Remove(3) and Decoy(5) can fail
    action_success = jnp.where(
        action_type == 3,  # Remove
        remove_success,
        jnp.where(
            action_type == 5,  # Decoy
            decoy_success,
            True  # All other actions succeed
        )
    )

    state = state.replace(last_blue_action_success=action_success)

    return state


def _apply_monitor(state: CageState, const: CageConst) -> CageState:
    """Monitor action: detect red activity on all hosts, clear unknown flags.

    CybORG behavior: Monitor detects process anomalies from Red's presence
    (malware, backdoors, shells). Scan activity creates network connections
    but not detectable processes. Only Exploit/PrivEsc create malware that
    Monitor can detect. The initial foothold is hidden (part of baseline).
    """
    # Detect recent exploit-level activity (Scan doesn't create detectable processes)
    recent_activity = state.red_activity_this_step >= ACTIVITY_EXPLOIT

    # Detect persistent Red presence (sessions), excluding initial foothold
    # This matches what observations show to Blue
    # Use zeros_like for JIT compatibility (can't use const.num_hosts directly in some contexts)
    has_red_presence = state.red_sessions > 0
    initial_foothold_mask = jnp.zeros_like(state.red_sessions, dtype=jnp.bool_)
    initial_foothold_mask = initial_foothold_mask.at[const.red_start_hosts].set(True)
    visible_presence = has_red_presence & ~initial_foothold_mask

    return state.replace(
        host_activity_detected=state.host_activity_detected | recent_activity | visible_presence,
        host_observation_unknown=jnp.zeros_like(state.host_observation_unknown),
    )


def _apply_analyse(state: CageState, target_host: int, const: CageConst) -> CageState:
    """Analyse action: detect activity and malware on target host, clear unknown flag.

    CybORG behavior: Analyse runs DensityScout which detects malware files
    (Density >= 0.9). This is the ONLY way Blue can see Privileged compromise.
    Also detects persistent Red presence (sessions). Scan doesn't create
    detectable malware, only Exploit/PrivEsc do.
    """
    # Detect recent exploit-level activity (Scan doesn't create malware)
    has_recent_activity = state.red_activity_this_step[target_host] >= ACTIVITY_EXPLOIT
    has_red_session = state.red_sessions[target_host] > 0

    # Check if this is the initial foothold (hidden from Blue)
    is_initial_foothold = jnp.any(const.red_start_hosts == target_host)
    visible_presence = has_red_session & ~is_initial_foothold

    should_detect = has_recent_activity | visible_presence
    new_detected = jnp.where(
        should_detect,
        state.host_activity_detected.at[target_host].set(True),
        state.host_activity_detected,
    )

    # Analyse detects malware via DensityScout - reveals Privileged compromise
    # Only Analyse can detect malware; PrivEsc sets host_has_malware but Blue
    # doesn't see Privileged until Analyse discovers it
    has_malware = state.host_has_malware[target_host]
    new_malware_detected = jnp.where(
        has_malware,
        state.host_malware_detected.at[target_host].set(True),
        state.host_malware_detected,
    )

    return state.replace(
        host_activity_detected=new_detected,
        host_observation_unknown=state.host_observation_unknown.at[target_host].set(False),
        host_malware_detected=new_malware_detected,
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
    new_valid_privesc = jax.lax.cond(
        is_initial_foothold,
        lambda: state.host_has_valid_privesc[target_host],
        lambda: jnp.array(False, dtype=jnp.bool_),
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
        host_has_malware=state.host_has_malware.at[target_host].set(False),
        host_malware_detected=state.host_malware_detected.at[target_host].set(False),
        host_has_valid_privesc=state.host_has_valid_privesc.at[target_host].set(new_valid_privesc),
    )


def _apply_decoy(state: CageState, target_host: int, decoy_type: int, const: CageConst) -> CageState:
    """Deploy decoy on target host if port available and OS compatible (matching CybORG).

    In CybORG, decoy deployment fails (returns success=FALSE) if:
    1. The port is already in use by an existing service
    2. The decoy requires a specific OS (Windows/Linux) that the host doesn't have

    We match this by only setting the decoy flag if both conditions pass.
    """
    # Check port conflict using DECOY_SERVICE_CONFLICTS
    conflict1 = DECOY_SERVICE_CONFLICTS[decoy_type, 0]
    conflict2 = DECOY_SERVICE_CONFLICTS[decoy_type, 1]
    has_conflict1 = jnp.where(conflict1 >= 0, state.host_services[target_host, conflict1], False)
    has_conflict2 = jnp.where(conflict2 >= 0, state.host_services[target_host, conflict2], False)
    port_available = ~has_conflict1 & ~has_conflict2

    # Check OS compatibility using DECOY_OS_ARRAY
    required_os = DECOY_OS_ARRAY[decoy_type]
    host_os = const.host_os[target_host]
    os_compatible = (required_os == OS_ANY) | (required_os == host_os)

    # Only deploy if port available AND OS compatible
    can_deploy = port_available & os_compatible
    new_decoy_value = jnp.where(can_deploy, True, state.host_decoys[target_host, decoy_type])

    # Add the decoy's service to host_services so Red can discover it via scan.
    # This mirrors CybORG's behavior where decoys create services with open ports.
    service_idx = DECOY_SERVICE_IDS[decoy_type]
    new_service_value = jnp.where(
        can_deploy,
        True,
        state.host_services[target_host, service_idx]
    )

    return state.replace(
        host_decoys=state.host_decoys.at[target_host, decoy_type].set(new_decoy_value),
        host_services=state.host_services.at[target_host, service_idx].set(new_service_value),
    )


def apply_red_action(
    state: CageState,
    action: chex.Array,
    const: CageConst,
    key: chex.PRNGKey
) -> CageState:
    """Apply red agent action to state.

    Also stores action info for ResilienceMetric:
    - last_red_action_type: 0=Sleep, 1=Discover, 2=Scan, 3=Exploit, 4=PrivEsc, 5=Impact
    - last_red_action_target: target host index (-1 for no target)
    - last_red_action_success: whether the action succeeded
    """
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
        lambda s: _apply_privesc(s, target_host, const, subkey),
        lambda s: s,
        state,
    )

    state = jax.lax.cond(
        action_type == 5,  # Impact
        lambda s: _apply_impact(s, target_host, const),
        lambda s: s,
        state,
    )

    # Store action info for ResilienceMetric calculation
    # target_host is valid for Scan(2), Exploit(3), PrivEsc(4), Impact(5)
    effective_target = jnp.where(action_type >= 2, target_host, -1)
    state = state.replace(
        last_red_action_type=action_type,
        last_red_action_target=effective_target,
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

    Note: CybORG does NOT require DiscoverSubnet before scanning. If Red has routing
    to a host (via session in adjacent subnet), they can scan it directly.
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
        red_activity_this_step=state.red_activity_this_step.at[target_host].set(
            jnp.where(can_scan, ACTIVITY_SCAN, state.red_activity_this_step[target_host])
        ),
        last_red_action_success=can_scan,
    )


# CybORG detection rate: 95% of exploits are detected by Blue (via Monitor/Analyse)
EXPLOIT_DETECTION_RATE = 0.95


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

    Detection: 95% of successful exploits generate detectable activity (matching CybORG).
    The remaining 5% succeed silently - Blue cannot detect them even with Monitor.

    CybORG behavior: Exploit requires prior DiscoverNetworkServices (scan) to know open ports.
    """
    num_hosts = const.num_hosts
    num_subnets = const.num_subnets
    target_subnet = const.host_subnet[target_host]

    # Check routing: Red needs session in adjacent subnet
    red_subnets = jnp.zeros(num_subnets, dtype=jnp.bool_)

    def check_subnet(i, red_subnets):
        has_session = state.red_sessions[i] > 0
        subnet = const.host_subnet[i]
        return red_subnets.at[subnet].set(red_subnets[subnet] | has_session)

    red_subnets = jax.lax.fori_loop(0, num_hosts, check_subnet, red_subnets)
    has_route = jnp.any(red_subnets & const.subnet_adjacency[:, target_subnet])

    services_on_host = state.host_services[target_host]
    exploit_vulnerabilities = const.service_exploits[:, exploit_type]
    has_vulnerable_service = jnp.any(services_on_host & exploit_vulnerabilities)

    # SSH (exploit_type 0) requires bruteforceable users on target host
    # CybORG's SSHBruteForce iterates through users and checks bruteforceable flag
    ssh_check = jnp.where(
        exploit_type == 0,  # SSHBruteForce
        const.bruteforceable_hosts[target_host],
        True  # non-SSH exploits don't need this check
    )

    # HTTP/HTTPS (exploit_type 2,3) requires RFI vulnerability on target host
    # CybORG's HTTPRFI checks "rfi" in process.properties
    rfi_check = jnp.where(
        (exploit_type == 2) | (exploit_type == 3),  # HTTPRFI or HTTPSRFI
        const.rfi_vulnerable_hosts[target_host],
        True  # non-HTTP exploits don't need this check
    )

    # Check if any blocking decoy is deployed on the target host
    # EXPLOIT_BLOCKED_BY_DECOYS[exploit_type] gives [exploit_idx, decoy1, decoy2]
    blocking_decoys = EXPLOIT_BLOCKED_BY_DECOYS[exploit_type]
    decoy1 = blocking_decoys[1]
    decoy2 = blocking_decoys[2]
    decoy1_present = jnp.where(decoy1 >= 0, state.host_decoys[target_host, decoy1], False)
    decoy2_present = jnp.where(decoy2 >= 0, state.host_decoys[target_host, decoy2], False)
    decoy_present = decoy1_present | decoy2_present

    # Decoy acts as honeypot - Red sees it as a valid target even without real service
    # This makes decoys useful on hosts that DON'T have the vulnerable service
    has_target = has_vulnerable_service | decoy_present

    # Exploit succeeds if: host is known (discovered OR scanned), real service exists,
    # no decoy blocks, AND exploit-specific preconditions are met
    # CybORG allows exploit if Red knows the host via DiscoverSubnet OR DiscoverNetworkServices
    host_known = state.red_discovered_hosts_jax[target_host] | state.red_scanned_hosts_jax[target_host]
    success = (host_known & has_route & has_vulnerable_service &
               ~decoy_present & ssh_check & rfi_check)

    # FTPDirectoryTraversal(1), HarakaRCE(4), SQLInjection(5), EternalBlue(6), BlueKeep(7) give root
    # Exception: BlueKeep on User2 gives NetworkService (user-level) due to RDP process user
    gives_root = (exploit_type == 1) | (exploit_type == 4) | (exploit_type == 5) | (exploit_type == 6) | (exploit_type == 7)
    bluekeep_on_user2 = (exploit_type == 7) & (target_host == const.user2_host_idx)
    gives_root = gives_root & ~bluekeep_on_user2

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

    # Detection: 95% of exploits are detectable (matching CybORG's detection_rate = 0.95)
    # 5% of exploits succeed silently and cannot be detected by Blue
    is_detected = jax.random.uniform(key) < EXPLOIT_DETECTION_RATE

    # Log activity when Red attempts exploit on a valid target (real service OR decoy)
    # This includes: successful exploit OR honeypot caught Red (decoy blocked)
    attempted_on_target = has_route & has_target
    activity_visible = attempted_on_target & is_detected

    # Set malware on successful exploit (CybORG creates cmd.exe/cmd.sh with density=0.9)
    new_malware = jnp.where(
        success,
        True,
        state.host_has_malware[target_host],
    )

    return state.replace(
        host_compromised=state.host_compromised.at[target_host].set(new_compromised),
        red_sessions=state.red_sessions.at[target_host].set(new_sessions),
        red_privilege=state.red_privilege.at[target_host].set(new_privilege),
        red_activity_this_step=state.red_activity_this_step.at[target_host].set(
            jnp.where(activity_visible, ACTIVITY_EXPLOIT, state.red_activity_this_step[target_host])
        ),
        host_has_malware=state.host_has_malware.at[target_host].set(new_malware),
        last_red_action_success=success,
    )


def _apply_privesc(state: CageState, target_host: int, const: CageConst, key: chex.PRNGKey) -> CageState:
    """Escalate privileges on target host.

    CybORG behavior:
    - If Red already has PRIVILEGED access, PrivEsc is a no-op (returns success but no change)
    - If Red has USER access, escalation succeeds and creates a valid privileged session
    - Valid privileged sessions count for confidentiality rewards
    - On Linux, the initial SYSTEM foothold doesn't count (it's an abstract session)
      but a PrivEsc from USER to root DOES count

    Also runs ExploreHost which discovers OTService on operational hosts, enabling Impact.
    """
    has_user_session = state.red_privilege[target_host] >= COMPROMISE_USER
    already_privileged = state.red_privilege[target_host] >= COMPROMISE_PRIVILEGED

    # PrivEsc succeeds if there's any user-level or higher session
    success = has_user_session

    # Only actually escalate if not already privileged
    new_privilege = jnp.where(
        success & ~already_privileged,
        COMPROMISE_PRIVILEGED,
        state.red_privilege[target_host],
    )

    new_compromised = jnp.where(
        success & ~already_privileged,
        COMPROMISE_PRIVILEGED,
        state.host_compromised[target_host],
    )

    # Malware is only created on actual escalation (not if already privileged)
    new_malware = jnp.where(
        success & ~already_privileged,
        True,
        state.host_has_malware[target_host],
    )

    # CybORG: PrivilegeEscalate runs ExploreHost which discovers OTService on operational hosts
    is_operational = const.operational_targets[target_host]
    discovers_ot = success & is_operational
    new_knows_ot = jnp.where(discovers_ot, True, state.red_knows_ot_service[target_host])

    # Valid privesc for rewards: only set when actual escalation happens (from USER)
    # This excludes cases where:
    # - The host was already privileged (initial foothold on Linux with SYSTEM = no escalation)
    # - The action failed
    actual_escalation = success & ~already_privileged
    new_valid_privesc = jnp.where(actual_escalation, True, state.host_has_valid_privesc[target_host])

    return state.replace(
        red_privilege=state.red_privilege.at[target_host].set(new_privilege),
        host_compromised=state.host_compromised.at[target_host].set(new_compromised),
        host_has_malware=state.host_has_malware.at[target_host].set(new_malware),
        red_knows_ot_service=state.red_knows_ot_service.at[target_host].set(new_knows_ot),
        host_has_valid_privesc=state.host_has_valid_privesc.at[target_host].set(new_valid_privesc),
        last_red_action_success=success,
    )


def _apply_impact(state: CageState, target_host: int, const: CageConst) -> CageState:
    """Impact action on operational host - stops OT service.

    CybORG behavior: Impact requires Red to have discovered the OT service first,
    which happens during PrivilegeEscalate's ExploreHost on operational hosts.
    """
    has_privileged = state.red_privilege[target_host] >= COMPROMISE_PRIVILEGED
    knows_ot_service = state.red_knows_ot_service[target_host]
    success = has_privileged & knows_ot_service

    new_ot_stopped = jnp.where(
        success,
        state.ot_service_stopped.at[target_host].set(True),
        state.ot_service_stopped,
    )

    return state.replace(
        ot_service_stopped=new_ot_stopped,
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
    def check_exploit_host(i, mask):
        def check_exploit_type(e, mask):
            exploit_action_idx = exploit_start + i * const.num_exploits + e
            exploit_valid = state.red_scanned_hosts_jax[i]
            return mask.at[exploit_action_idx].set(exploit_valid)

        return jax.lax.fori_loop(0, const.num_exploits, check_exploit_type, mask)

    mask = jax.lax.fori_loop(0, const.num_hosts, check_exploit_host, mask)

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


def decode_blue_action_name(action: int, const: CageConst) -> str:
    """Convert blue action index to human-readable name like 'Analyse@Enterprise1'.

    This is a Python function for logging/debugging, not for use in JIT-compiled code.
    """
    action_type, target_host, decoy_type = decode_blue_action(action, const)
    action_type = int(action_type)
    target_host = int(target_host)
    decoy_type = int(decoy_type)

    type_name = BLUE_ACTION_NAMES[action_type] if action_type < len(BLUE_ACTION_NAMES) else f"Unknown({action_type})"

    if action_type in (0, 1):  # Sleep, Monitor
        return type_name
    elif action_type == 5:  # Decoy
        host_name = HOST_NAMES.get(target_host, f"Host{target_host}")
        decoy_name = DECOY_NAMES.get(decoy_type, f"Decoy{decoy_type}")
        return f"{decoy_name}@{host_name}"
    else:  # Analyse, Remove, Restore
        host_name = HOST_NAMES.get(target_host, f"Host{target_host}")
        return f"{type_name}@{host_name}"


def decode_red_action_name(action: int, const: CageConst) -> str:
    """Convert red action index to human-readable name like 'ExploitRemoteService(SSHBruteForce)@User0'.

    This is a Python function for logging/debugging, not for use in JIT-compiled code.
    """
    action_type, target_subnet, target_host, exploit_type = decode_red_action(action, const)
    action_type = int(action_type)
    target_subnet = int(target_subnet)
    target_host = int(target_host)
    exploit_type = int(exploit_type)

    type_name = RED_ACTION_NAMES[action_type] if action_type < len(RED_ACTION_NAMES) else f"Unknown({action_type})"

    if action_type == 0:  # Sleep
        return type_name
    elif action_type == 1:  # DiscoverRemoteSystems
        subnet_name = SUBNET_NAMES.get(target_subnet, f"Subnet{target_subnet}")
        return f"{type_name}@{subnet_name}"
    elif action_type == 2:  # DiscoverNetworkServices
        host_name = HOST_NAMES.get(target_host, f"Host{target_host}")
        return f"{type_name}@{host_name}"
    elif action_type == 3:  # Exploit
        host_name = HOST_NAMES.get(target_host, f"Host{target_host}")
        exploit_name = EXPLOIT_NAMES.get(exploit_type, f"Exploit{exploit_type}")
        return f"{type_name}({exploit_name})@{host_name}"
    else:  # PrivilegeEscalate, Impact
        host_name = HOST_NAMES.get(target_host, f"Host{target_host}")
        return f"{type_name}@{host_name}"
