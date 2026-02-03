"""Bidirectional action translation between CybORG objects and JAX indices.

This module handles the mapping between CybORG's action objects and the
integer indices used by CAGE-JAX.
"""

from typing import Dict, Optional, Tuple

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
from jaxmarl.environments.cage.config import ScenarioConfig

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

DECOY_NAME_TO_IDX = {
    'DecoyApache': 0, 'DecoyFemitter': 1, 'DecoyHarakaSMPT': 2,
    'DecoySmss': 3, 'DecoySSHD': 4, 'DecoySvchost': 5,
    'DecoyTomcat': 6, 'DecoyVsftpd': 7,
}


def get_host_mappings(config: Optional[ScenarioConfig] = None) -> Tuple[Dict[str, int], Dict[int, str], int]:
    """Get host ID mappings from config or use defaults.

    Returns:
        Tuple of (host_ids, host_names, num_hosts)
    """
    if config is not None:
        return config.host_ids, config.host_names, config.num_hosts
    return HOST_IDS, HOST_NAMES, NUM_HOSTS


def get_subnet_mappings(config: Optional[ScenarioConfig] = None, const: Optional[CageConst] = None) -> Tuple[Dict[str, int], Dict[int, str], int]:
    """Get subnet ID mappings from config or use defaults.

    Returns:
        Tuple of (subnet_name_to_idx, idx_to_subnet_name, num_subnets)
    """
    if config is not None:
        subnet_name_to_idx = config.subnet_ids
        idx_to_subnet_name = {v: k for k, v in subnet_name_to_idx.items()}
        return subnet_name_to_idx, idx_to_subnet_name, config.num_subnets

    return SUBNET_IDS, {v: k for k, v in SUBNET_IDS.items()}, len(SUBNET_IDS)


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


def cyborg_blue_action_to_jax(
    action,
    cyborg_env,
    const: Optional[CageConst] = None,
    config: Optional[ScenarioConfig] = None,
) -> int:
    """Convert CybORG blue action object to JAX action index.

    Args:
        action: CybORG action object
        cyborg_env: CybORG environment for IP resolution
        const: Optional CageConst for action offset calculation
        config: Optional ScenarioConfig for host mappings

    Returns:
        JAX action index
    """
    host_ids, host_names, num_hosts = get_host_mappings(config)
    class_name = type(action).__name__

    if class_name == 'Sleep':
        return BLUE_SLEEP

    if class_name == 'Monitor':
        return BLUE_MONITOR

    if class_name == 'Analyse':
        hostname = getattr(action, 'hostname', None)
        if hostname and hostname in host_ids:
            return BLUE_ANALYSE_START + host_ids[hostname]
        return BLUE_SLEEP

    if class_name == 'Remove':
        hostname = getattr(action, 'hostname', None)
        if hostname and hostname in host_ids:
            return BLUE_REMOVE_START + host_ids[hostname]
        return BLUE_SLEEP

    if class_name == 'Restore':
        hostname = getattr(action, 'hostname', None)
        if hostname and hostname in host_ids:
            return BLUE_RESTORE_START + host_ids[hostname]
        return BLUE_SLEEP

    if class_name.startswith('Decoy'):
        hostname = getattr(action, 'hostname', None)
        if hostname and hostname in host_ids and class_name in DECOY_NAME_TO_IDX:
            host_idx = host_ids[hostname]
            decoy_type = DECOY_NAME_TO_IDX[class_name]
            return BLUE_DECOY_START + decoy_type * num_hosts + host_idx
        return BLUE_SLEEP

    return BLUE_SLEEP


def cyborg_red_action_to_jax(
    action,
    cyborg_env,
    const: Optional[CageConst] = None,
    use_sub_action: bool = True,
    config: Optional[ScenarioConfig] = None,
) -> int:
    """Convert CybORG red action object to JAX action index.

    Args:
        action: CybORG action object
        cyborg_env: CybORG environment for IP resolution
        const: Optional CageConst for action offset calculation
        use_sub_action: If True, use sub_action for ExploitRemoteService
        config: Optional ScenarioConfig for host mappings

    Returns:
        JAX action index
    """
    host_ids, host_names, num_hosts = get_host_mappings(config)
    subnet_name_to_idx, idx_to_subnet_name, num_subnets = get_subnet_mappings(config, const)
    class_name = type(action).__name__

    if class_name == 'Sleep':
        return RED_SLEEP

    if class_name == 'DiscoverRemoteSystems':
        subnet = getattr(action, 'subnet', None)
        if subnet:
            subnet_str = str(subnet)

            import re
            cidr_match = re.search(r'(\d+\.\d+\.\d+)\.\d+/\d+', subnet_str)
            if cidr_match:
                cidr_prefix = cidr_match.group(1)
                ip_map = cyborg_env.get_ip_map()
                for hostname, host_ip in ip_map.items():
                    ip_str = str(host_ip)
                    host_prefix = '.'.join(ip_str.split('.')[:3])
                    if host_prefix == cidr_prefix and hostname in host_ids:
                        if hostname.startswith('User'):
                            return RED_DISCOVER_SUBNET_START + subnet_name_to_idx.get('User', 0)
                        elif hostname.startswith('Enterprise') or hostname == 'Defender':
                            return RED_DISCOVER_SUBNET_START + subnet_name_to_idx.get('Enterprise', 1)
                        elif hostname.startswith('Op_'):
                            return RED_DISCOVER_SUBNET_START + subnet_name_to_idx.get('Operational', 2)

            if hasattr(subnet, 'value'):
                subnet_name = subnet.value if isinstance(subnet.value, str) else str(subnet)
            else:
                subnet_name = subnet_str

            subnet_name = subnet_name.split('.')[-1].replace("IPv4Network('", '').replace("')", '')

            if subnet_name in subnet_name_to_idx:
                return RED_DISCOVER_SUBNET_START + subnet_name_to_idx[subnet_name]
            for name, idx in subnet_name_to_idx.items():
                if name.lower() in subnet_name.lower():
                    return RED_DISCOVER_SUBNET_START + idx
        return RED_SLEEP

    if class_name == 'DiscoverNetworkServices':
        ip = getattr(action, 'ip_address', None)
        if ip:
            ip_map = cyborg_env.get_ip_map()
            for hostname, host_ip in ip_map.items():
                if str(host_ip) == str(ip) and hostname in host_ids:
                    return RED_SCAN_HOST_START + host_ids[hostname]
        return RED_SLEEP

    if class_name == 'ExploitRemoteService':
        ip = getattr(action, 'ip_address', None)
        host_idx = None

        if ip:
            ip_map = cyborg_env.get_ip_map()
            for hostname, host_ip in ip_map.items():
                if str(host_ip) == str(ip) and hostname in host_ids:
                    host_idx = host_ids[hostname]
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
            return RED_EXPLOIT_START + host_idx * len(EXPLOIT_CLASS_TO_JAX_IDX) + exploit_idx

        return RED_SLEEP

    if class_name in EXPLOIT_CLASS_TO_JAX_IDX:
        ip = getattr(action, 'ip_address', None)
        host_idx = None

        if ip:
            ip_map = cyborg_env.get_ip_map()
            for hostname, host_ip in ip_map.items():
                if str(host_ip) == str(ip) and hostname in host_ids:
                    host_idx = host_ids[hostname]
                    break

        if host_idx is None:
            return RED_SLEEP

        exploit_idx = EXPLOIT_CLASS_TO_JAX_IDX[class_name]
        return RED_EXPLOIT_START + host_idx * len(EXPLOIT_CLASS_TO_JAX_IDX) + exploit_idx

    if class_name == 'PrivilegeEscalate':
        hostname = getattr(action, 'hostname', None)
        if hostname and hostname in host_ids:
            return RED_PRIVESC_START + host_ids[hostname]
        return RED_SLEEP

    if class_name == 'Impact':
        hostname = getattr(action, 'hostname', None)
        if hostname and hostname in host_ids:
            return RED_IMPACT_START + host_ids[hostname]
        return RED_SLEEP

    return RED_SLEEP


def cyborg_action_to_jax(
    action,
    cyborg_env,
    agent: str,
    const: Optional[CageConst] = None,
    config: Optional[ScenarioConfig] = None,
) -> int:
    """Convert CybORG action to JAX action index.

    Args:
        action: CybORG action object
        cyborg_env: CybORG environment for IP resolution
        agent: 'Blue' or 'Red'
        const: Optional CageConst for action offset calculation
        config: Optional ScenarioConfig for host mappings

    Returns:
        JAX action index
    """
    if agent.lower() == 'blue':
        return cyborg_blue_action_to_jax(action, cyborg_env, const, config)
    return cyborg_red_action_to_jax(action, cyborg_env, const, config=config)


def jax_blue_action_to_cyborg(
    action_idx: int,
    cyborg_env,
    config: Optional[ScenarioConfig] = None,
):
    """Convert JAX blue action index to CybORG action object.

    Args:
        action_idx: JAX action index
        cyborg_env: CybORG environment for action space access
        config: Optional ScenarioConfig for host mappings

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

    host_ids, host_names, num_hosts = get_host_mappings(config)

    if action_idx == BLUE_SLEEP:
        return Sleep()

    if action_idx == BLUE_MONITOR:
        return Monitor(session=0, agent='Blue')

    if BLUE_ANALYSE_START <= action_idx < BLUE_REMOVE_START:
        host_idx = action_idx - BLUE_ANALYSE_START
        hostname = host_names.get(host_idx)
        if hostname:
            return Analyse(session=0, agent='Blue', hostname=hostname)
        return Sleep()

    if BLUE_REMOVE_START <= action_idx < BLUE_DECOY_START:
        host_idx = action_idx - BLUE_REMOVE_START
        hostname = host_names.get(host_idx)
        if hostname:
            return Remove(session=0, agent='Blue', hostname=hostname)
        return Sleep()

    if BLUE_DECOY_START <= action_idx < BLUE_RESTORE_START:
        offset = action_idx - BLUE_DECOY_START
        decoy_idx = offset // num_hosts
        host_idx = offset % num_hosts
        hostname = host_names.get(host_idx)
        if hostname and decoy_idx < len(DECOY_CLASSES):
            decoy_class = DECOY_CLASSES[decoy_idx]
            return decoy_class(session=0, agent='Blue', hostname=hostname)
        return Sleep()

    if action_idx >= BLUE_RESTORE_START:
        host_idx = action_idx - BLUE_RESTORE_START
        hostname = host_names.get(host_idx)
        if hostname:
            return Restore(session=0, agent='Blue', hostname=hostname)
        return Sleep()

    return Sleep()


def jax_red_action_to_cyborg(
    action_idx: int,
    cyborg_env,
    known_ips: dict = None,
    config: Optional[ScenarioConfig] = None,
):
    """Convert JAX red action index to CybORG action object.

    Args:
        action_idx: JAX action index
        cyborg_env: CybORG environment for action space access
        known_ips: Optional dict of hostname -> IP that Red has discovered.
                   If provided, uses these IPs for scan/exploit actions.
                   If None, falls back to get_ip_map().
        config: Optional ScenarioConfig for host mappings

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

    host_ids, host_names, num_hosts = get_host_mappings(config)
    subnet_name_to_idx, idx_to_subnet_name, num_subnets = get_subnet_mappings(config)

    if action_idx == RED_SLEEP:
        return Sleep()

    if RED_DISCOVER_SUBNET_START <= action_idx < RED_SCAN_HOST_START:
        subnet_idx = action_idx - RED_DISCOVER_SUBNET_START
        subnet_name = idx_to_subnet_name.get(subnet_idx, 'User')
        state = cyborg_env.environment_controller.state
        subnet_cidr = state.subnet_name_to_cidr.get(subnet_name)
        return DiscoverRemoteSystems(session=0, agent='Red', subnet=subnet_cidr)

    if RED_SCAN_HOST_START <= action_idx < RED_EXPLOIT_START:
        host_idx = action_idx - RED_SCAN_HOST_START
        hostname = host_names.get(host_idx)
        if hostname:
            if known_ips and hostname in known_ips:
                ip = known_ips[hostname]
            else:
                ip = cyborg_env.get_ip_map().get(hostname)
            return DiscoverNetworkServices(session=0, agent='Red', ip_address=ip)
        return Sleep()

    if RED_EXPLOIT_START <= action_idx < RED_PRIVESC_START:
        offset = action_idx - RED_EXPLOIT_START
        num_exploits = len(EXPLOIT_CLASSES)
        host_idx = offset // num_exploits
        exploit_idx = offset % num_exploits
        hostname = host_names.get(host_idx)

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
        hostname = host_names.get(host_idx)
        if hostname:
            return PrivilegeEscalate(session=0, agent='Red', hostname=hostname)
        return Sleep()

    if action_idx >= RED_IMPACT_START:
        host_idx = action_idx - RED_IMPACT_START
        hostname = host_names.get(host_idx)
        if hostname:
            return Impact(session=0, agent='Red', hostname=hostname)
        return Sleep()

    return Sleep()


def jax_action_to_cyborg(
    action_idx: int,
    cyborg_env,
    agent: str,
    known_ips: dict = None,
    config: Optional[ScenarioConfig] = None,
):
    """Convert JAX action index to CybORG action object.

    Args:
        action_idx: JAX action index
        cyborg_env: CybORG environment for action space access
        agent: 'Blue' or 'Red'
        known_ips: Optional dict of hostname -> IP that Red has discovered.
        config: Optional ScenarioConfig for host mappings

    Returns:
        CybORG action object
    """
    if agent.lower() == 'blue':
        return jax_blue_action_to_cyborg(action_idx, cyborg_env, config)
    return jax_red_action_to_cyborg(action_idx, cyborg_env, known_ips, config)


def describe_jax_blue_action(
    action_idx: int,
    config: Optional[ScenarioConfig] = None,
) -> str:
    """Get human-readable description of JAX blue action."""
    host_ids, host_names, num_hosts = get_host_mappings(config)

    if action_idx == BLUE_SLEEP:
        return "Sleep"

    if action_idx == BLUE_MONITOR:
        return "Monitor"

    if BLUE_ANALYSE_START <= action_idx < BLUE_REMOVE_START:
        host_idx = action_idx - BLUE_ANALYSE_START
        hostname = host_names.get(host_idx, f"host_{host_idx}")
        return f"Analyse({hostname})"

    if BLUE_REMOVE_START <= action_idx < BLUE_DECOY_START:
        host_idx = action_idx - BLUE_REMOVE_START
        hostname = host_names.get(host_idx, f"host_{host_idx}")
        return f"Remove({hostname})"

    if BLUE_DECOY_START <= action_idx < BLUE_RESTORE_START:
        offset = action_idx - BLUE_DECOY_START
        decoy_idx = offset // num_hosts
        host_idx = offset % num_hosts
        hostname = host_names.get(host_idx, f"host_{host_idx}")
        decoy_names = list(DECOY_NAME_TO_IDX.keys())
        decoy_name = decoy_names[decoy_idx] if decoy_idx < len(decoy_names) else f"decoy_{decoy_idx}"
        return f"{decoy_name}({hostname})"

    if action_idx >= BLUE_RESTORE_START:
        host_idx = action_idx - BLUE_RESTORE_START
        hostname = host_names.get(host_idx, f"host_{host_idx}")
        return f"Restore({hostname})"

    return f"Unknown({action_idx})"


def describe_jax_red_action(
    action_idx: int,
    config: Optional[ScenarioConfig] = None,
    const: Optional[CageConst] = None,
) -> str:
    """Get human-readable description of JAX red action."""
    host_ids, host_names, num_hosts = get_host_mappings(config)
    subnet_name_to_idx, idx_to_subnet_name, num_subnets = get_subnet_mappings(config, const)

    if action_idx == RED_SLEEP:
        return "Sleep"

    if RED_DISCOVER_SUBNET_START <= action_idx < RED_SCAN_HOST_START:
        subnet_idx = action_idx - RED_DISCOVER_SUBNET_START
        subnet_name = idx_to_subnet_name.get(subnet_idx, f"subnet_{subnet_idx}")
        return f"DiscoverSubnet({subnet_name})"

    if RED_SCAN_HOST_START <= action_idx < RED_EXPLOIT_START:
        host_idx = action_idx - RED_SCAN_HOST_START
        hostname = host_names.get(host_idx, f"host_{host_idx}")
        return f"Scan({hostname})"

    if RED_EXPLOIT_START <= action_idx < RED_PRIVESC_START:
        offset = action_idx - RED_EXPLOIT_START
        num_exploits = len(EXPLOIT_CLASS_TO_JAX_IDX)
        host_idx = offset // num_exploits
        exploit_idx = offset % num_exploits
        hostname = host_names.get(host_idx, f"host_{host_idx}")
        exploit_name = JAX_IDX_TO_EXPLOIT_CLASS.get(exploit_idx, f"exploit_{exploit_idx}")
        return f"{exploit_name}({hostname})"

    if RED_PRIVESC_START <= action_idx < RED_IMPACT_START:
        host_idx = action_idx - RED_PRIVESC_START
        hostname = host_names.get(host_idx, f"host_{host_idx}")
        return f"PrivEsc({hostname})"

    if action_idx >= RED_IMPACT_START:
        host_idx = action_idx - RED_IMPACT_START
        hostname = host_names.get(host_idx, f"host_{host_idx}")
        return f"Impact({hostname})"

    return f"Unknown({action_idx})"
