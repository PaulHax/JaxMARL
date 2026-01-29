"""Predefined action sequences and scenarios for comparison tests.

Contains killchain sequences, exploit test patterns, and blue defense patterns
used in systematic testing.
"""

from typing import List, Dict

from jaxmarl.environments.cage.state import HOST_IDS, build_const_from_config
from jaxmarl.environments.cage.actions import (
    RED_SLEEP, RED_DISCOVER_SUBNET_START, RED_SCAN_HOST_START,
    RED_EXPLOIT_START, RED_PRIVESC_START, RED_IMPACT_START,
    BLUE_SLEEP, BLUE_MONITOR, BLUE_REMOVE_START, BLUE_RESTORE_START,
    BLUE_ANALYSE_START, BLUE_DECOY_START, NUM_DECOY_TYPES,
    NUM_HOSTS, NUM_EXPLOITS,
)
from jaxmarl.environments.cage.cyborg_loader import get_scenario_from_cyborg

_config = get_scenario_from_cyborg('Scenario2')
_const = build_const_from_config(_config)

SUBNET_USER = int(_const.host_subnet[HOST_IDS['User0']])
SUBNET_ENTERPRISE = int(_const.host_subnet[HOST_IDS['Enterprise0']])
SUBNET_OPERATIONAL = int(_const.host_subnet[HOST_IDS['Op_Server0']])

EXPLOIT_SSH = 0
EXPLOIT_FTP = 1
EXPLOIT_HTTP = 2
EXPLOIT_HTTPS = 3
EXPLOIT_HARAKA = 4
EXPLOIT_SQL = 5
EXPLOIT_ETERNAL = 6
EXPLOIT_BLUEKEEP = 7

DECOY_APACHE = 0
DECOY_FEMITTER = 1
DECOY_HARAKA = 2
DECOY_SMSS = 3
DECOY_SSHD = 4
DECOY_SVCHOST = 5
DECOY_TOMCAT = 6
DECOY_VSFTPD = 7


def red_discover_subnet(subnet: int) -> int:
    """Get action index for DiscoverRemoteSystems on subnet."""
    return RED_DISCOVER_SUBNET_START + subnet


def red_scan_host(hostname: str) -> int:
    """Get action index for DiscoverNetworkServices on host."""
    return RED_SCAN_HOST_START + HOST_IDS[hostname]


def red_exploit_host(hostname: str, exploit_type: int) -> int:
    """Get action index for Exploit on host with specific exploit."""
    return RED_EXPLOIT_START + HOST_IDS[hostname] * NUM_EXPLOITS + exploit_type


def red_privesc_host(hostname: str) -> int:
    """Get action index for PrivilegeEscalate on host."""
    return RED_PRIVESC_START + HOST_IDS[hostname]


def red_impact_host(hostname: str) -> int:
    """Get action index for Impact on host."""
    return RED_IMPACT_START + HOST_IDS[hostname]


def blue_analyse_host(hostname: str) -> int:
    """Get action index for Analyse on host."""
    return BLUE_ANALYSE_START + HOST_IDS[hostname]


def blue_remove_host(hostname: str) -> int:
    """Get action index for Remove on host."""
    return BLUE_REMOVE_START + HOST_IDS[hostname]


def blue_restore_host(hostname: str) -> int:
    """Get action index for Restore on host."""
    return BLUE_RESTORE_START + HOST_IDS[hostname]


def blue_decoy_host(hostname: str, decoy_type: int) -> int:
    """Get action index for deploying decoy on host."""
    return BLUE_DECOY_START + HOST_IDS[hostname] * NUM_DECOY_TYPES + decoy_type


BLINE_KILLCHAIN_STANDARD = [
    red_discover_subnet(SUBNET_USER),
    red_scan_host('User1'),
    red_exploit_host('User1', EXPLOIT_SSH),
    red_privesc_host('User1'),
    red_discover_subnet(SUBNET_ENTERPRISE),
    red_scan_host('Enterprise1'),
    red_exploit_host('Enterprise1', EXPLOIT_SSH),
    red_privesc_host('Enterprise1'),
    red_discover_subnet(SUBNET_OPERATIONAL),
    red_scan_host('Enterprise2'),
    red_exploit_host('Enterprise2', EXPLOIT_SSH),
    red_privesc_host('Enterprise2'),
    red_scan_host('Op_Server0'),
    red_exploit_host('Op_Server0', EXPLOIT_SSH),
    red_privesc_host('Op_Server0'),
    red_impact_host('Op_Server0'),
]


KILLCHAIN_VIA_ENTERPRISE0 = [
    red_discover_subnet(SUBNET_USER),
    red_scan_host('User4'),
    red_exploit_host('User4', EXPLOIT_HARAKA),  # Haraka gives root access on User hosts
    red_privesc_host('User4'),
    red_scan_host('Enterprise0'),  # No DiscoverSubnet needed - can reach from User
    red_exploit_host('Enterprise0', EXPLOIT_SSH),
    red_privesc_host('Enterprise0'),
    red_discover_subnet(SUBNET_ENTERPRISE),
    red_scan_host('Enterprise2'),
    red_exploit_host('Enterprise2', EXPLOIT_HTTPS),  # B_line uses HTTPSRFI on Enterprise2
    red_privesc_host('Enterprise2'),
    red_scan_host('Op_Server0'),
    red_exploit_host('Op_Server0', EXPLOIT_SSH),
    red_privesc_host('Op_Server0'),
    red_impact_host('Op_Server0'),
]


KILLCHAIN_VIA_HARAKA = [
    red_discover_subnet(SUBNET_USER),
    red_scan_host('User3'),
    red_exploit_host('User3', EXPLOIT_HARAKA),
    red_privesc_host('User3'),
    red_scan_host('Enterprise0'),  # Only Enterprise0 is reachable from User subnet
    red_exploit_host('Enterprise0', EXPLOIT_SSH),
    red_privesc_host('Enterprise0'),
    red_discover_subnet(SUBNET_ENTERPRISE),
    red_scan_host('Enterprise2'),
    red_exploit_host('Enterprise2', EXPLOIT_SSH),
    red_privesc_host('Enterprise2'),
    red_scan_host('Op_Server0'),
    red_exploit_host('Op_Server0', EXPLOIT_SSH),
    red_privesc_host('Op_Server0'),
    red_impact_host('Op_Server0'),
]


def generate_single_exploit_sequence(hostname: str, exploit_type: int) -> List[int]:
    """Generate sequence to exploit a single host.

    Returns action sequence: discover subnet -> scan host -> exploit
    """
    host_subnet = get_host_subnet(hostname)
    prereq_subnet = get_prereq_subnet(host_subnet)

    actions = []
    if prereq_subnet is not None and prereq_subnet != SUBNET_USER:
        actions.extend(get_path_to_subnet(prereq_subnet))

    actions.extend([
        red_discover_subnet(host_subnet),
        red_scan_host(hostname),
        red_exploit_host(hostname, exploit_type),
    ])
    return actions


def get_host_subnet(hostname: str) -> int:
    """Get subnet index for hostname."""
    if hostname.startswith('User'):
        return SUBNET_USER
    elif hostname.startswith('Enterprise') or hostname == 'Defender':
        return SUBNET_ENTERPRISE
    else:
        return SUBNET_OPERATIONAL


def get_prereq_subnet(target_subnet: int) -> int:
    """Get subnet that must be compromised to reach target."""
    if target_subnet == SUBNET_USER:
        return None
    elif target_subnet == SUBNET_ENTERPRISE:
        return SUBNET_USER
    else:
        return SUBNET_ENTERPRISE


def get_path_to_subnet(target_subnet: int) -> List[int]:
    """Get action sequence to establish access to subnet."""
    if target_subnet == SUBNET_USER:
        return []
    elif target_subnet == SUBNET_ENTERPRISE:
        return [
            red_discover_subnet(SUBNET_USER),
            red_scan_host('User1'),
            red_exploit_host('User1', EXPLOIT_SSH),
            red_privesc_host('User1'),
        ]
    else:
        return [
            red_discover_subnet(SUBNET_USER),
            red_scan_host('User1'),
            red_exploit_host('User1', EXPLOIT_SSH),
            red_privesc_host('User1'),
            red_discover_subnet(SUBNET_ENTERPRISE),
            red_scan_host('Enterprise1'),
            red_exploit_host('Enterprise1', EXPLOIT_SSH),
            red_privesc_host('Enterprise1'),
        ]


ALL_HOST_EXPLOIT_SEQUENCES: Dict[str, List[int]] = {}
for host in HOST_IDS.keys():
    if host != 'User0':
        ALL_HOST_EXPLOIT_SEQUENCES[host] = generate_single_exploit_sequence(host, EXPLOIT_SSH)


def exploit_test_sequence(hostname: str, exploit_types: List[int]) -> List[List[int]]:
    """Generate sequences testing multiple exploit types on one host.

    Args:
        hostname: Target host
        exploit_types: List of exploit type indices to test

    Returns:
        List of action sequences, one per exploit type
    """
    base_path = generate_single_exploit_sequence(hostname, EXPLOIT_SSH)[:-1]
    sequences = []

    for exploit_type in exploit_types:
        seq = base_path.copy()
        seq.append(red_exploit_host(hostname, exploit_type))
        sequences.append(seq)

    return sequences


BLUE_DEFENSIVE_MONITOR_ALL = [BLUE_MONITOR] * 100

BLUE_DECOY_ENTERPRISE = [
    blue_decoy_host('Enterprise0', DECOY_SSHD),
    blue_decoy_host('Enterprise1', DECOY_SSHD),
    blue_decoy_host('Enterprise2', DECOY_SSHD),
] + [BLUE_MONITOR] * 97

BLUE_REACTIVE_REMOVE_TEMPLATE = {
    'trigger_host': 'Enterprise1',
    'action_fn': lambda hostname: blue_remove_host(hostname),
}


SCENARIO_CONFIGS = {
    'passive_blue': {
        'blue_actions': [BLUE_SLEEP] * 100,
        'description': 'Blue does nothing',
    },
    'monitor_only': {
        'blue_actions': BLUE_DEFENSIVE_MONITOR_ALL,
        'description': 'Blue only monitors',
    },
    'decoy_defense': {
        'blue_actions': BLUE_DECOY_ENTERPRISE,
        'description': 'Blue deploys SSH decoys on Enterprise hosts',
    },
}


def get_exploit_host_pairs() -> List[tuple]:
    """Get all valid (hostname, exploit_type) pairs for testing."""
    pairs = []

    user_hosts = ['User1', 'User2', 'User3', 'User4']
    enterprise_hosts = ['Enterprise0', 'Enterprise1', 'Enterprise2', 'Defender']
    op_hosts = ['Op_Server0', 'Op_Host0', 'Op_Host1', 'Op_Host2']

    for host in user_hosts + enterprise_hosts + op_hosts:
        for exploit in range(8):
            pairs.append((host, exploit))

    return pairs
