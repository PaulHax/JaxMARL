"""Bidirectional action translation between CybORG objects and JAX indices.

This module handles the mapping between CybORG's action objects and the
integer indices used by CAGE-JAX.
"""

from typing import Optional, Type, Union
import jax.numpy as jnp

from jaxmarl.environments.cage.state import (
    HOST_IDS, HOST_NAMES, SUBNET_IDS,
    CageConst, NUM_HOSTS,
)
from jaxmarl.environments.cage.actions import (
    BLUE_SLEEP, BLUE_MONITOR, BLUE_ANALYSE_START, BLUE_REMOVE_START,
    BLUE_DECOY_START, BLUE_RESTORE_START, NUM_DECOY_TYPES,
    RED_SLEEP, RED_DISCOVER_SUBNET_START, RED_SCAN_HOST_START,
    RED_EXPLOIT_START, RED_PRIVESC_START, RED_IMPACT_START,
    get_blue_action_offsets, get_red_action_offsets,
)

EXPLOIT_CLASS_TO_JAX_IDX = {
    'SSHBruteForce': 0,
    'FTPDirectoryTraversal': 1,
    'HTTPRFI': 2,
    'HTTPSRFI': 3,
    'HarakaRCE': 4,
    'SQLInjection': 5,
    'EternalBlue': 6,
    'BlueKeep': 7,
}

JAX_IDX_TO_EXPLOIT_CLASS = {v: k for k, v in EXPLOIT_CLASS_TO_JAX_IDX.items()}

from jaxmarl.environments.cage.cyborg_loader import get_scenario_from_cyborg
from jaxmarl.environments.cage.state import build_const_from_config

_config = get_scenario_from_cyborg('Scenario2')
_const = build_const_from_config(_config)

SUBNET_NAME_TO_IDX = {
    'User': int(_const.host_subnet[HOST_IDS['User0']]),
    'Enterprise': int(_const.host_subnet[HOST_IDS['Enterprise0']]),
    'Operational': int(_const.host_subnet[HOST_IDS['Op_Server0']]),
}
IDX_TO_SUBNET_NAME = {v: k for k, v in SUBNET_NAME_TO_IDX.items()}

# Map IP ranges to subnet indices (based on Scenario2 layout)
# User subnet: 10.0.140.x
# Enterprise subnet: 10.0.57.x
# Operational subnet: 10.0.12.x
CIDR_TO_SUBNET_IDX = {
    '10.0.140': 0,  # User
    '10.0.57': 1,   # Enterprise
    '10.0.12': 2,   # Operational
}

DECOY_NAME_TO_IDX = {
    'DecoyApache': 0, 'DecoyFemitter': 1, 'DecoyHarakaSMPT': 2,
    'DecoySmss': 3, 'DecoySSHD': 4, 'DecoySvchost': 5,
    'DecoyTomcat': 6, 'DecoyVsftpd': 7,
}


def _find_session_on_host(cyborg_env, hostname: str) -> int:
    """Find Red's session ID on the given host.

    Returns the session ID if Red has a session on the host, otherwise 0.
    """
    try:
        state = cyborg_env.environment_controller.state
        sessions = state.sessions.get('Red', {})
        for session_id, session in sessions.items():
            if session.host == hostname:
                return session_id
    except Exception:
        pass
    return 0


def get_exploit_class_from_action(action) -> Optional[str]:
    """Extract the exploit class name from a CybORG action.

    CybORG's ExploitRemoteService dynamically selects exploits based on ports.
    After execution, inspect action.sub_action to get the actual exploit class.

    Args:
        action: CybORG action object (may have sub_action attribute)

    Returns:
        Exploit class name string or None if not an exploit action
    """
    if hasattr(action, 'sub_action') and action.sub_action is not None:
        return type(action.sub_action).__name__

    class_name = type(action).__name__
    if class_name in EXPLOIT_CLASS_TO_JAX_IDX:
        return class_name

    return None


def cyborg_blue_action_to_jax(action, cyborg_env, const: Optional[CageConst] = None) -> int:
    """Convert CybORG blue action object to JAX action index.

    Args:
        action: CybORG action object
        cyborg_env: CybORG environment for IP resolution
        const: Optional CageConst for action offset calculation

    Returns:
        JAX action index
    """
    class_name = type(action).__name__

    if class_name == 'Sleep':
        return BLUE_SLEEP

    if class_name == 'Monitor':
        return BLUE_MONITOR

    if class_name == 'Analyse':
        hostname = getattr(action, 'hostname', None)
        if hostname and hostname in HOST_IDS:
            return BLUE_ANALYSE_START + HOST_IDS[hostname]
        return BLUE_SLEEP

    if class_name == 'Remove':
        hostname = getattr(action, 'hostname', None)
        if hostname and hostname in HOST_IDS:
            return BLUE_REMOVE_START + HOST_IDS[hostname]
        return BLUE_SLEEP

    if class_name == 'Restore':
        hostname = getattr(action, 'hostname', None)
        if hostname and hostname in HOST_IDS:
            return BLUE_RESTORE_START + HOST_IDS[hostname]
        return BLUE_SLEEP

    if class_name.startswith('Decoy'):
        hostname = getattr(action, 'hostname', None)
        if hostname and hostname in HOST_IDS and class_name in DECOY_NAME_TO_IDX:
            host_idx = HOST_IDS[hostname]
            decoy_type = DECOY_NAME_TO_IDX[class_name]
            # CybORG encoding: decoy_type * num_hosts + host_idx
            return BLUE_DECOY_START + decoy_type * NUM_HOSTS + host_idx
        return BLUE_SLEEP

    return BLUE_SLEEP


def cyborg_red_action_to_jax(
    action,
    cyborg_env,
    const: Optional[CageConst] = None,
    use_sub_action: bool = True,
) -> int:
    """Convert CybORG red action object to JAX action index.

    Args:
        action: CybORG action object
        cyborg_env: CybORG environment for IP resolution
        const: Optional CageConst for action offset calculation
        use_sub_action: If True, use sub_action for ExploitRemoteService

    Returns:
        JAX action index
    """
    class_name = type(action).__name__

    if class_name == 'Sleep':
        return RED_SLEEP

    if class_name == 'DiscoverRemoteSystems':
        subnet = getattr(action, 'subnet', None)
        if subnet:
            subnet_str = str(subnet)

            # Extract IP prefix from CIDR (e.g., "10.0.140" from "10.0.140.208/28")
            import re
            cidr_match = re.search(r'(\d+\.\d+\.\d+)\.\d+/\d+', subnet_str)
            if cidr_match:
                cidr_prefix = cidr_match.group(1)
                # Find which subnet this IP belongs to by checking host IPs
                ip_map = cyborg_env.get_ip_map()
                for hostname, host_ip in ip_map.items():
                    ip_str = str(host_ip)
                    host_prefix = '.'.join(ip_str.split('.')[:3])
                    if host_prefix == cidr_prefix and hostname in HOST_IDS:
                        # Found a host in this subnet - determine subnet from hostname
                        if hostname.startswith('User'):
                            return RED_DISCOVER_SUBNET_START + SUBNET_NAME_TO_IDX['User']
                        elif hostname.startswith('Enterprise') or hostname == 'Defender':
                            return RED_DISCOVER_SUBNET_START + SUBNET_NAME_TO_IDX['Enterprise']
                        elif hostname.startswith('Op_'):
                            return RED_DISCOVER_SUBNET_START + SUBNET_NAME_TO_IDX['Operational']

            # Fall back to name matching
            if hasattr(subnet, 'value'):
                subnet_name = subnet.value if isinstance(subnet.value, str) else str(subnet)
            else:
                subnet_name = subnet_str

            subnet_name = subnet_name.split('.')[-1].replace("IPv4Network('", '').replace("')", '')

            if subnet_name in SUBNET_NAME_TO_IDX:
                return RED_DISCOVER_SUBNET_START + SUBNET_NAME_TO_IDX[subnet_name]
            for name, idx in SUBNET_NAME_TO_IDX.items():
                if name.lower() in subnet_name.lower():
                    return RED_DISCOVER_SUBNET_START + idx
        return RED_SLEEP

    if class_name == 'DiscoverNetworkServices':
        ip = getattr(action, 'ip_address', None)
        if ip:
            ip_map = cyborg_env.get_ip_map()
            for hostname, host_ip in ip_map.items():
                if str(host_ip) == str(ip) and hostname in HOST_IDS:
                    return RED_SCAN_HOST_START + HOST_IDS[hostname]
        return RED_SLEEP

    if class_name == 'ExploitRemoteService':
        ip = getattr(action, 'ip_address', None)
        host_idx = None

        if ip:
            ip_map = cyborg_env.get_ip_map()
            for hostname, host_ip in ip_map.items():
                if str(host_ip) == str(ip) and hostname in HOST_IDS:
                    host_idx = HOST_IDS[hostname]
                    break

        if host_idx is None:
            return RED_SLEEP

        exploit_name = None
        if use_sub_action and hasattr(action, 'sub_action') and action.sub_action is not None:
            exploit_name = type(action.sub_action).__name__
        else:
            exploit_name = 'SSHBruteForce'

        if exploit_name in EXPLOIT_CLASS_TO_JAX_IDX:
            exploit_idx = EXPLOIT_CLASS_TO_JAX_IDX[exploit_name]
            # Exploit encoding: host * num_exploits + exploit_type
            return RED_EXPLOIT_START + host_idx * len(EXPLOIT_CLASS_TO_JAX_IDX) + exploit_idx

        return RED_SLEEP

    if class_name in EXPLOIT_CLASS_TO_JAX_IDX:
        ip = getattr(action, 'ip_address', None)
        host_idx = None

        if ip:
            ip_map = cyborg_env.get_ip_map()
            for hostname, host_ip in ip_map.items():
                if str(host_ip) == str(ip) and hostname in HOST_IDS:
                    host_idx = HOST_IDS[hostname]
                    break

        if host_idx is None:
            return RED_SLEEP

        exploit_idx = EXPLOIT_CLASS_TO_JAX_IDX[class_name]
        # Exploit encoding: host * num_exploits + exploit_type
        return RED_EXPLOIT_START + host_idx * len(EXPLOIT_CLASS_TO_JAX_IDX) + exploit_idx

    if class_name == 'PrivilegeEscalate':
        hostname = getattr(action, 'hostname', None)
        if hostname and hostname in HOST_IDS:
            return RED_PRIVESC_START + HOST_IDS[hostname]
        return RED_SLEEP

    if class_name == 'Impact':
        hostname = getattr(action, 'hostname', None)
        if hostname and hostname in HOST_IDS:
            return RED_IMPACT_START + HOST_IDS[hostname]
        return RED_SLEEP

    return RED_SLEEP


def cyborg_action_to_jax(
    action,
    cyborg_env,
    agent: str,
    const: Optional[CageConst] = None,
) -> int:
    """Convert CybORG action to JAX action index.

    Args:
        action: CybORG action object
        cyborg_env: CybORG environment for IP resolution
        agent: 'Blue' or 'Red'
        const: Optional CageConst for action offset calculation

    Returns:
        JAX action index
    """
    if agent.lower() == 'blue':
        return cyborg_blue_action_to_jax(action, cyborg_env, const)
    return cyborg_red_action_to_jax(action, cyborg_env, const)


def jax_blue_action_to_cyborg(action_idx: int, cyborg_env):
    """Convert JAX blue action index to CybORG action object.

    Args:
        action_idx: JAX action index
        cyborg_env: CybORG environment for action space access

    Returns:
        CybORG action object
    """
    from CybORG.Shared.Actions import (
        Sleep, Monitor, Analyse, Remove, Restore,
        DecoyApache, DecoyFemitter, DecoyHarakaSMPT, DecoySmss,
        DecoySSHD, DecoySvchost, DecoyTomcat, DecoyVsftpd,
    )

    DECOY_CLASSES = [
        DecoyApache, DecoyFemitter, DecoyHarakaSMPT, DecoySmss,
        DecoySSHD, DecoySvchost, DecoyTomcat, DecoyVsftpd,
    ]

    if action_idx == BLUE_SLEEP:
        return Sleep()

    if action_idx == BLUE_MONITOR:
        return Monitor(session=0, agent='Blue')

    if BLUE_ANALYSE_START <= action_idx < BLUE_REMOVE_START:
        host_idx = action_idx - BLUE_ANALYSE_START
        hostname = HOST_NAMES.get(host_idx)
        if hostname:
            return Analyse(session=0, agent='Blue', hostname=hostname)
        return Sleep()

    if BLUE_REMOVE_START <= action_idx < BLUE_DECOY_START:
        host_idx = action_idx - BLUE_REMOVE_START
        hostname = HOST_NAMES.get(host_idx)
        if hostname:
            return Remove(session=0, agent='Blue', hostname=hostname)
        return Sleep()

    if BLUE_DECOY_START <= action_idx < BLUE_RESTORE_START:
        offset = action_idx - BLUE_DECOY_START
        # CybORG encoding: decoy_type * num_hosts + host_idx
        decoy_idx = offset // NUM_HOSTS
        host_idx = offset % NUM_HOSTS
        hostname = HOST_NAMES.get(host_idx)
        if hostname and decoy_idx < len(DECOY_CLASSES):
            decoy_class = DECOY_CLASSES[decoy_idx]
            return decoy_class(session=0, agent='Blue', hostname=hostname)
        return Sleep()

    if action_idx >= BLUE_RESTORE_START:
        host_idx = action_idx - BLUE_RESTORE_START
        hostname = HOST_NAMES.get(host_idx)
        if hostname:
            return Restore(session=0, agent='Blue', hostname=hostname)
        return Sleep()

    return Sleep()


def jax_red_action_to_cyborg(action_idx: int, cyborg_env, known_ips: dict = None):
    """Convert JAX red action index to CybORG action object.

    Args:
        action_idx: JAX action index
        cyborg_env: CybORG environment for action space access
        known_ips: Optional dict of hostname -> IP that Red has discovered.
                   If provided, uses these IPs for scan/exploit actions.
                   If None, falls back to get_ip_map().

    Returns:
        CybORG action object
    """
    from CybORG.Shared.Actions import (
        Sleep, DiscoverRemoteSystems, DiscoverNetworkServices,
        ExploitRemoteService, PrivilegeEscalate, Impact,
        SSHBruteForce, FTPDirectoryTraversal, HTTPRFI, HTTPSRFI,
        HarakaRCE, SQLInjection, EternalBlue, BlueKeep,
    )

    EXPLOIT_CLASSES = [
        SSHBruteForce, FTPDirectoryTraversal, HTTPRFI, HTTPSRFI,
        HarakaRCE, SQLInjection, EternalBlue, BlueKeep,
    ]

    if action_idx == RED_SLEEP:
        return Sleep()

    if RED_DISCOVER_SUBNET_START <= action_idx < RED_SCAN_HOST_START:
        subnet_idx = action_idx - RED_DISCOVER_SUBNET_START
        subnet_name = IDX_TO_SUBNET_NAME.get(subnet_idx, 'User')
        state = cyborg_env.environment_controller.state
        subnet_cidr = state.subnet_name_to_cidr.get(subnet_name)
        return DiscoverRemoteSystems(session=0, agent='Red', subnet=subnet_cidr)

    if RED_SCAN_HOST_START <= action_idx < RED_EXPLOIT_START:
        host_idx = action_idx - RED_SCAN_HOST_START
        hostname = HOST_NAMES.get(host_idx)
        if hostname:
            if known_ips and hostname in known_ips:
                ip = known_ips[hostname]
            else:
                ip = cyborg_env.get_ip_map().get(hostname)
            return DiscoverNetworkServices(session=0, agent='Red', ip_address=ip)
        return Sleep()

    if RED_EXPLOIT_START <= action_idx < RED_PRIVESC_START:
        offset = action_idx - RED_EXPLOIT_START
        # Encoding: host * num_exploits + exploit_type
        num_exploits = len(EXPLOIT_CLASSES)
        host_idx = offset // num_exploits
        exploit_idx = offset % num_exploits
        hostname = HOST_NAMES.get(host_idx)

        if hostname and exploit_idx < len(EXPLOIT_CLASSES):
            if known_ips and hostname in known_ips:
                ip = known_ips[hostname]
            else:
                ip = cyborg_env.get_ip_map().get(hostname)
            exploit_class = EXPLOIT_CLASSES[exploit_idx]
            return exploit_class(ip_address=ip, agent='Red', session=0, target_session=0)
        return Sleep()

    if RED_PRIVESC_START <= action_idx < RED_IMPACT_START:
        host_idx = action_idx - RED_PRIVESC_START
        hostname = HOST_NAMES.get(host_idx)
        if hostname:
            return PrivilegeEscalate(session=0, agent='Red', hostname=hostname)
        return Sleep()

    if action_idx >= RED_IMPACT_START:
        host_idx = action_idx - RED_IMPACT_START
        hostname = HOST_NAMES.get(host_idx)
        if hostname:
            return Impact(session=0, agent='Red', hostname=hostname)
        return Sleep()

    return Sleep()


def jax_action_to_cyborg(action_idx: int, cyborg_env, agent: str, known_ips: dict = None):
    """Convert JAX action index to CybORG action object.

    Args:
        action_idx: JAX action index
        cyborg_env: CybORG environment for action space access
        agent: 'Blue' or 'Red'
        known_ips: Optional dict of hostname -> IP that Red has discovered.

    Returns:
        CybORG action object
    """
    if agent.lower() == 'blue':
        return jax_blue_action_to_cyborg(action_idx, cyborg_env)
    return jax_red_action_to_cyborg(action_idx, cyborg_env, known_ips)


def describe_jax_blue_action(action_idx: int) -> str:
    """Get human-readable description of JAX blue action."""
    if action_idx == BLUE_SLEEP:
        return "Sleep"

    if action_idx == BLUE_MONITOR:
        return "Monitor"

    if BLUE_ANALYSE_START <= action_idx < BLUE_REMOVE_START:
        host_idx = action_idx - BLUE_ANALYSE_START
        hostname = HOST_NAMES.get(host_idx, f"host_{host_idx}")
        return f"Analyse({hostname})"

    if BLUE_REMOVE_START <= action_idx < BLUE_DECOY_START:
        host_idx = action_idx - BLUE_REMOVE_START
        hostname = HOST_NAMES.get(host_idx, f"host_{host_idx}")
        return f"Remove({hostname})"

    if BLUE_DECOY_START <= action_idx < BLUE_RESTORE_START:
        offset = action_idx - BLUE_DECOY_START
        # CybORG encoding: decoy_type * num_hosts + host_idx
        decoy_idx = offset // NUM_HOSTS
        host_idx = offset % NUM_HOSTS
        hostname = HOST_NAMES.get(host_idx, f"host_{host_idx}")
        decoy_names = list(DECOY_NAME_TO_IDX.keys())
        decoy_name = decoy_names[decoy_idx] if decoy_idx < len(decoy_names) else f"decoy_{decoy_idx}"
        return f"{decoy_name}({hostname})"

    if action_idx >= BLUE_RESTORE_START:
        host_idx = action_idx - BLUE_RESTORE_START
        hostname = HOST_NAMES.get(host_idx, f"host_{host_idx}")
        return f"Restore({hostname})"

    return f"Unknown({action_idx})"


def describe_jax_red_action(action_idx: int) -> str:
    """Get human-readable description of JAX red action."""
    if action_idx == RED_SLEEP:
        return "Sleep"

    if RED_DISCOVER_SUBNET_START <= action_idx < RED_SCAN_HOST_START:
        subnet_idx = action_idx - RED_DISCOVER_SUBNET_START
        subnet_name = IDX_TO_SUBNET_NAME.get(subnet_idx, f"subnet_{subnet_idx}")
        return f"DiscoverSubnet({subnet_name})"

    if RED_SCAN_HOST_START <= action_idx < RED_EXPLOIT_START:
        host_idx = action_idx - RED_SCAN_HOST_START
        hostname = HOST_NAMES.get(host_idx, f"host_{host_idx}")
        return f"Scan({hostname})"

    if RED_EXPLOIT_START <= action_idx < RED_PRIVESC_START:
        offset = action_idx - RED_EXPLOIT_START
        num_exploits = len(EXPLOIT_CLASS_TO_JAX_IDX)
        host_idx = offset // num_exploits
        exploit_idx = offset % num_exploits
        hostname = HOST_NAMES.get(host_idx, f"host_{host_idx}")
        exploit_name = JAX_IDX_TO_EXPLOIT_CLASS.get(exploit_idx, f"exploit_{exploit_idx}")
        return f"{exploit_name}({hostname})"

    if RED_PRIVESC_START <= action_idx < RED_IMPACT_START:
        host_idx = action_idx - RED_PRIVESC_START
        hostname = HOST_NAMES.get(host_idx, f"host_{host_idx}")
        return f"PrivEsc({hostname})"

    if action_idx >= RED_IMPACT_START:
        host_idx = action_idx - RED_IMPACT_START
        hostname = HOST_NAMES.get(host_idx, f"host_{host_idx}")
        return f"Impact({hostname})"

    return f"Unknown({action_idx})"
