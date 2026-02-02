"""CAGE-JAX configurable scenario definitions.

This module provides configuration structures for creating CAGE environments
with variable numbers of hosts, subnets, and network topologies.

Scenario2 configuration is loaded directly from CybORG YAML files to ensure
consistency with the reference implementation. See cyborg_loader.py for details.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import jax.numpy as jnp
import chex
import CybORG

CYBORG_PATH = Path(CybORG.__file__).parent
CYBORG_SCENARIOS_PATH = CYBORG_PATH / 'Shared' / 'Scenarios'
CYBORG_IMAGES_PATH = CYBORG_SCENARIOS_PATH / 'images'

# OS type constants
OS_LINUX = 0
OS_WINDOWS = 1
OS_ANY = -1  # Compatible with any OS

# Decoy OS restrictions: decoy_idx -> required_os (OS_LINUX, OS_WINDOWS, or OS_ANY)
# DecoySmss, DecoySvchost, DecoyFemitter = Windows only
# DecoyVsftpd, DecoyHarakaSMPT = Linux only
DECOY_OS_RESTRICTIONS = {
    'DecoyApache': OS_ANY,
    'DecoyFemitter': OS_WINDOWS,
    'DecoyHarakaSMPT': OS_LINUX,
    'DecoySmss': OS_WINDOWS,
    'DecoySSHD': OS_ANY,
    'DecoySvchost': OS_WINDOWS,
    'DecoyTomcat': OS_ANY,
    'DecoyVsftpd': OS_LINUX,
}


@dataclass
class HostConfig:
    """Configuration for a single host."""
    name: str
    subnet: str
    os: str = "linux"  # "linux" or "windows"
    confidentiality: float = 0.1  # reward weight
    availability: float = 0.0  # reward weight
    services: List[str] = field(default_factory=list)
    service_properties: Dict[str, List[str]] = field(default_factory=dict)  # e.g. {'http': ['rfi']}
    is_operational_target: bool = False  # for Impact action
    has_bruteforceable_users: bool = False  # for SSHBruteForce


@dataclass
class SubnetConfig:
    """Configuration for a subnet."""
    name: str
    hosts: List[str] = field(default_factory=list)
    connected_subnets: List[str] = field(default_factory=list)


@dataclass
class AgentConfig:
    """Configuration for an agent (Red or Blue)."""
    name: str
    starting_host: Optional[str] = None
    team: str = "blue"  # "blue" or "red"


@dataclass
class ScenarioConfig:
    """Complete scenario configuration."""
    name: str
    hosts: List[HostConfig]
    subnets: List[SubnetConfig]
    agents: List[AgentConfig]
    max_steps: int = 100

    # Service and exploit definitions (shared across scenarios)
    services: List[str] = field(default_factory=lambda: [
        'ssh', 'ftp', 'http', 'https', 'smtp', 'mysql', 'smb', 'rdp', 'tomcat', 'haraka'
    ])
    exploits: List[str] = field(default_factory=lambda: [
        'SSHBruteForce', 'FTPDirectoryTraversal', 'HTTPRFI', 'HTTPSRFI',
        'HarakaRCE', 'SQLInjection', 'EternalBlue', 'BlueKeep'
    ])
    decoy_types: List[str] = field(default_factory=lambda: [
        'DecoyApache', 'DecoyFemitter', 'DecoyHarakaSMPT', 'DecoySmss',
        'DecoySSHD', 'DecoySvchost', 'DecoyTomcat', 'DecoyVsftpd'
    ])

    # Service-exploit vulnerability mapping (service_idx -> list of exploit_idx)
    service_vulnerabilities: Dict[str, List[str]] = field(default_factory=lambda: {
        'ssh': ['SSHBruteForce'],
        'ftp': ['FTPDirectoryTraversal'],
        'http': ['HTTPRFI'],
        'https': ['HTTPSRFI'],
        'haraka': ['HarakaRCE'],
        'mysql': ['SQLInjection'],
        'smb': ['EternalBlue'],
        'rdp': ['BlueKeep'],
    })

    @property
    def num_hosts(self) -> int:
        return len(self.hosts)

    @property
    def num_subnets(self) -> int:
        return len(self.subnets)

    @property
    def num_services(self) -> int:
        return len(self.services)

    @property
    def num_exploits(self) -> int:
        return len(self.exploits)

    @property
    def num_decoys(self) -> int:
        return len(self.decoy_types)

    @property
    def host_ids(self) -> Dict[str, int]:
        return {h.name: i for i, h in enumerate(self.hosts)}

    @property
    def host_names(self) -> Dict[int, str]:
        return {i: h.name for i, h in enumerate(self.hosts)}

    @property
    def subnet_ids(self) -> Dict[str, int]:
        return {s.name: i for i, s in enumerate(self.subnets)}

    @property
    def service_ids(self) -> Dict[str, int]:
        return {s: i for i, s in enumerate(self.services)}

    @property
    def exploit_ids(self) -> Dict[str, int]:
        return {e: i for i, e in enumerate(self.exploits)}

    @property
    def decoy_ids(self) -> Dict[str, int]:
        return {d: i for i, d in enumerate(self.decoy_types)}

    def get_host_subnet(self, host_name: str) -> int:
        """Get subnet index for a host."""
        for h in self.hosts:
            if h.name == host_name:
                return self.subnet_ids[h.subnet]
        raise ValueError(f"Unknown host: {host_name}")

    def get_red_agents(self) -> List[AgentConfig]:
        """Get all red team agents."""
        return [a for a in self.agents if a.team == "red"]

    def get_blue_agents(self) -> List[AgentConfig]:
        """Get all blue team agents."""
        return [a for a in self.agents if a.team == "blue"]

    @property
    def num_red_agents(self) -> int:
        return len(self.get_red_agents())

    @property
    def num_blue_agents(self) -> int:
        return len(self.get_blue_agents())

    @property
    def decoy_host_indices(self) -> List[int]:
        """Get indices of hosts that can have decoys (all hosts, matching CybORG)."""
        return list(range(len(self.hosts)))

    @property
    def operational_target_indices(self) -> List[int]:
        """Get indices of hosts that are operational targets (for Impact rewards)."""
        return [i for i, h in enumerate(self.hosts) if h.is_operational_target]


def create_scenario2_config() -> ScenarioConfig:
    """Create CAGE Challenge 2 Scenario 2 configuration.

    Loads configuration directly from CybORG YAML files to ensure consistency
    with the reference implementation.
    """
    from jaxmarl.environments.cage.cyborg_loader import load_scenario_from_yaml
    return load_scenario_from_yaml(
        CYBORG_SCENARIOS_PATH / 'Scenario2.yaml',
        CYBORG_IMAGES_PATH,
    )


def create_scalable_config(
    num_users: int = 5,
    num_enterprise: int = 3,
    num_op_hosts: int = 3,
    num_op_servers: int = 1,
    name: str = 'ScalableScenario',
    max_steps: int = 100,
) -> ScenarioConfig:
    """Create a scalable scenario configuration.

    Args:
        num_users: Number of User hosts
        num_enterprise: Number of Enterprise hosts (excluding Defender)
        num_op_hosts: Number of operational workstation hosts
        num_op_servers: Number of operational server hosts
        name: Scenario name
        max_steps: Maximum episode steps

    Returns:
        ScenarioConfig with the specified topology
    """
    hosts = []

    # User hosts
    for i in range(num_users):
        conf = 0.0 if i == 0 else 0.1
        hosts.append(HostConfig(f'User{i}', 'User', 'linux', confidentiality=conf, availability=0.0, services=['ssh']))

    # Enterprise hosts
    enterprise_services = [['ssh', 'http'], ['ssh', 'smb'], ['ssh', 'tomcat']]
    for i in range(num_enterprise):
        svcs = enterprise_services[i % len(enterprise_services)]
        hosts.append(HostConfig(f'Enterprise{i}', 'Enterprise', 'windows', confidentiality=1.0, availability=1.0, services=svcs))

    # Defender
    hosts.append(HostConfig('Defender', 'Enterprise', 'linux', confidentiality=0.1, availability=0.1, services=['ssh']))

    # Operational hosts
    for i in range(num_op_hosts):
        os_type = 'windows' if i % 3 == 2 else 'linux'
        svcs = ['rdp'] if os_type == 'windows' else ['ssh']
        hosts.append(HostConfig(f'Op_Host{i}', 'Operational', os_type, confidentiality=0.1, availability=0.1, services=svcs))

    # Operational servers
    for i in range(num_op_servers):
        hosts.append(HostConfig(f'Op_Server{i}', 'Operational', 'windows', confidentiality=1.0, availability=10.0, services=['ssh', 'http'], is_operational_target=True))

    # Build subnet host lists
    user_hosts = [f'User{i}' for i in range(num_users)]
    enterprise_hosts = [f'Enterprise{i}' for i in range(num_enterprise)] + ['Defender']
    op_hosts = [f'Op_Host{i}' for i in range(num_op_hosts)] + [f'Op_Server{i}' for i in range(num_op_servers)]

    subnets = [
        SubnetConfig('User', user_hosts, ['User', 'Enterprise']),
        SubnetConfig('Enterprise', enterprise_hosts, ['User', 'Enterprise', 'Operational']),
        SubnetConfig('Operational', op_hosts, ['Enterprise', 'Operational']),
    ]

    agents = [
        AgentConfig('Blue', starting_host='Defender', team='blue'),
        AgentConfig('Red', starting_host='User0', team='red'),
    ]

    return ScenarioConfig(
        name=name,
        hosts=hosts,
        subnets=subnets,
        agents=agents,
        max_steps=max_steps,
    )


def create_hosts_2_config() -> ScenarioConfig:
    """Create hosts_2 scalability scenario (approximately 2x base hosts)."""
    return create_scalable_config(
        num_users=10,
        num_enterprise=6,
        num_op_hosts=7,
        num_op_servers=1,
        name='hosts_2',
    )


def create_hosts_3_config() -> ScenarioConfig:
    """Create hosts_3 scalability scenario (approximately 3x base hosts)."""
    return create_scalable_config(
        num_users=15,
        num_enterprise=9,
        num_op_hosts=11,
        num_op_servers=1,
        name='hosts_3',
    )


def create_hosts_4_config() -> ScenarioConfig:
    """Create hosts_4 scalability scenario (approximately 4x base hosts)."""
    return create_scalable_config(
        num_users=20,
        num_enterprise=12,
        num_op_hosts=15,
        num_op_servers=1,
        name='hosts_4',
    )


def create_hosts_5_config() -> ScenarioConfig:
    """Create hosts_5 scalability scenario (approximately 5x base hosts)."""
    return create_scalable_config(
        num_users=25,
        num_enterprise=16,
        num_op_hosts=19,
        num_op_servers=1,
        name='hosts_5',
    )


def create_multi_agent_config(
    num_red_agents: int = 2,
    num_blue_agents: int = 2,
    base_config: Optional[ScenarioConfig] = None,
) -> ScenarioConfig:
    """Create a multi-agent scenario configuration.

    Args:
        num_red_agents: Number of red team agents
        num_blue_agents: Number of blue team agents
        base_config: Base scenario to extend (default: Scenario2)

    Returns:
        ScenarioConfig with multiple agents
    """
    if base_config is None:
        base_config = create_scenario2_config()

    # Get available starting positions
    user_hosts = [h.name for h in base_config.hosts if h.subnet == 'User']

    agents = []

    # Blue agents
    for i in range(num_blue_agents):
        agents.append(AgentConfig(f'Blue{i}', starting_host='Defender', team='blue'))

    # Red agents (each starts on a different user host if possible)
    for i in range(num_red_agents):
        start_host = user_hosts[i % len(user_hosts)]
        agents.append(AgentConfig(f'Red{i}', starting_host=start_host, team='red'))

    return ScenarioConfig(
        name=f'{base_config.name}_multi_{num_red_agents}r_{num_blue_agents}b',
        hosts=base_config.hosts,
        subnets=base_config.subnets,
        agents=agents,
        max_steps=base_config.max_steps,
        services=base_config.services,
        exploits=base_config.exploits,
        decoy_types=base_config.decoy_types,
        service_vulnerabilities=base_config.service_vulnerabilities,
    )


# Registry of available scenarios
SCENARIOS = {
    'Scenario2': create_scenario2_config,
    'hosts_2': create_hosts_2_config,
    'hosts_3': create_hosts_3_config,
    'hosts_4': create_hosts_4_config,
    'hosts_5': create_hosts_5_config,
}


def get_scenario(name: str) -> ScenarioConfig:
    """Get scenario configuration by name."""
    if name not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {name}. Available: {list(SCENARIOS.keys())}")
    return SCENARIOS[name]()
