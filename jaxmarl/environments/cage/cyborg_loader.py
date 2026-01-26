"""Load JaxMARL scenario configuration from CybORG YAML files.

This module provides functions to parse CybORG's YAML scenario and image files
and convert them to JaxMARL's ScenarioConfig format. This ensures JaxMARL
stays in sync with CybORG's scenario definitions without manual duplication.
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import yaml

from jaxmarl.environments.cage.config import (
    HostConfig, SubnetConfig, AgentConfig, ScenarioConfig
)


# Port-to-service mapping (port alone is usually sufficient)
PORT_TO_SERVICE = {
    22: 'ssh',
    21: 'ftp',
    80: 'http',
    443: 'https',
    25: 'smtp',
    139: 'smb',
    445: 'smb',
}

# Port-to-service mapping that requires process type confirmation
PORT_PROCESS_TYPE_TO_SERVICE = {
    (3389, 'rdp'): 'rdp',
}

# Process name to service mapping (for processes without standard ports)
PROCESS_NAME_TO_SERVICE = {
    'mysql': 'mysql',
    'tomcat8.exe': 'tomcat',
}

# Confidentiality value mapping from CybORG strings to numeric values
CONFIDENTIALITY_MAP = {
    'None': 0.0,
    'Low': 0.1,
    'Medium': 1.0,
    'High': 1.0,
}

# Availability value mapping from CybORG strings to numeric values
AVAILABILITY_MAP = {
    'None': 0.0,
    'Low': 0.1,
    'Medium': 1.0,
    'High': 10.0,
}


def find_cyborg_path() -> Optional[Path]:
    """Find the CybORG installation path by looking for it relative to common locations."""
    search_paths = [
        Path(__file__).parent.parent.parent.parent.parent / 'cage-challenge-2' / 'CybORG' / 'CybORG',
        Path.home() / 'src' / 'cyber' / 'cage-challenge-2' / 'CybORG' / 'CybORG',
        Path('/home/paulhax/src/cyber/cage-challenge-2/CybORG/CybORG'),
    ]

    for path in search_paths:
        if path.exists() and (path / 'Shared' / 'Scenarios').exists():
            return path

    return None


def load_yaml_file(file_path: Path) -> dict:
    """Load a YAML file and return its contents as a dictionary."""
    with open(file_path) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def extract_services_from_processes(processes: List[dict]) -> List[str]:
    """Extract service names from CybORG process definitions.

    Services are identified by:
    1. Port number (e.g., port 22 → ssh)
    2. Process name (e.g., mysql → mysql)
    3. Process Type for services like RDP
    4. Process Version for haraka detection
    """
    services = set()

    for process in processes:
        connections = process.get('Connections', [])
        process_name = process.get('Process Name', '').lower()
        process_version = process.get('Process Version', '')
        process_type = process.get('Process Type', '').lower()

        # Check connections for port-based services
        for conn in connections:
            port = conn.get('local_port')
            if port in PORT_TO_SERVICE:
                service = PORT_TO_SERVICE[port]
                # Special case: port 25 with haraka version
                if port == 25 and 'haraka' in process_version.lower():
                    services.add('haraka')
                else:
                    services.add(service)

            # Check port + process type combinations
            key = (port, process_type)
            if key in PORT_PROCESS_TYPE_TO_SERVICE:
                services.add(PORT_PROCESS_TYPE_TO_SERVICE[key])

        # Check process name for additional services
        for proc_pattern, service in PROCESS_NAME_TO_SERVICE.items():
            if proc_pattern.lower() in process_name:
                services.add(service)

    return sorted(services)


def extract_os_type(system_info: dict) -> str:
    """Extract OS type from CybORG system info."""
    os_type = system_info.get('OSType', 'LINUX')
    return 'windows' if os_type.upper() == 'WINDOWS' else 'linux'


def load_scenario_from_cyborg(
    scenario_name: str = 'Scenario2',
    cyborg_path: Optional[Path] = None,
) -> ScenarioConfig:
    """Load scenario configuration by parsing CybORG YAML files.

    Args:
        scenario_name: Name of the scenario (e.g., 'Scenario2')
        cyborg_path: Optional path to CybORG installation. If not provided,
                     attempts to find it automatically.

    Returns:
        ScenarioConfig populated from CybORG's YAML files.

    Raises:
        FileNotFoundError: If CybORG path or scenario files cannot be found.
    """
    if cyborg_path is None:
        cyborg_path = find_cyborg_path()
        if cyborg_path is None:
            raise FileNotFoundError(
                "Could not find CybORG installation. Please provide cyborg_path."
            )

    scenarios_path = cyborg_path / 'Shared' / 'Scenarios'
    images_path = scenarios_path / 'images'

    # Load main scenario file
    scenario_file = scenarios_path / f'{scenario_name}.yaml'
    if not scenario_file.exists():
        raise FileNotFoundError(f"Scenario file not found: {scenario_file}")

    scenario_data = load_yaml_file(scenario_file)

    # Load images mapping
    images_yaml = load_yaml_file(images_path / 'images.yaml')

    # Parse hosts
    hosts_data = scenario_data.get('Hosts', {})
    subnets_data = scenario_data.get('Subnets', {})
    agents_data = scenario_data.get('Agents', {})

    # Build subnet-to-hosts mapping
    subnet_hosts = {}
    for subnet_name, subnet_info in subnets_data.items():
        subnet_hosts[subnet_name] = subnet_info.get('Hosts', [])

    # Build host configs
    host_configs = []
    for host_name in sorted(hosts_data.keys()):
        host_info = hosts_data[host_name]
        image_name = host_info.get('image')

        # Find which subnet this host belongs to
        host_subnet = None
        for subnet_name, hosts_in_subnet in subnet_hosts.items():
            if host_name in hosts_in_subnet:
                host_subnet = subnet_name
                break

        if host_subnet is None:
            continue

        # Load image YAML
        image_mapping = images_yaml.get(image_name, {})
        image_yaml_path = image_mapping.get('path')

        if image_yaml_path:
            image_file = images_path / f'{image_yaml_path}.yaml'
            image_data = load_yaml_file(image_file)
            host_image_data = image_data.get('Test_Host', {})
        else:
            host_image_data = {}

        # Extract services from processes
        processes = host_image_data.get('Processes', [])
        services = extract_services_from_processes(processes)

        # Extract OS type
        system_info = host_image_data.get('System info', {})
        os_type = extract_os_type(system_info)

        # Extract confidentiality/availability values
        conf_value = host_info.get('ConfidentialityValue', 'Low')
        avail_value = host_info.get('AvailabilityValue', 'Low')
        confidentiality = CONFIDENTIALITY_MAP.get(conf_value, 0.1)
        availability = AVAILABILITY_MAP.get(avail_value, 0.0)

        # Check if operational target (has OTService)
        is_operational_target = False
        cyborg_services = host_image_data.get('Services', {})
        if 'OTService' in cyborg_services:
            is_operational_target = True

        host_configs.append(HostConfig(
            name=host_name,
            subnet=host_subnet,
            os=os_type,
            confidentiality=confidentiality,
            availability=availability,
            services=services,
            is_operational_target=is_operational_target,
        ))

    # Build subnet configs with connectivity
    subnet_configs = []
    for subnet_name in subnets_data.keys():
        subnet_info = subnets_data[subnet_name]
        hosts_in_subnet = [h.name for h in host_configs if h.subnet == subnet_name]

        # Determine connected subnets from NACLs
        nacls = subnet_info.get('NACLs', {})
        connected = set()
        connected.add(subnet_name)

        for target, rules in nacls.items():
            if target == 'all':
                connected.update(subnets_data.keys())
            elif target in subnets_data:
                out_rule = rules.get('out', 'None')
                if out_rule == 'all' or out_rule != 'None':
                    connected.add(target)

        subnet_configs.append(SubnetConfig(
            name=subnet_name,
            hosts=hosts_in_subnet,
            connected_subnets=sorted(connected),
        ))

    # Build agent configs
    agent_configs = []
    for agent_name, agent_info in agents_data.items():
        if agent_name in ('Green',):
            continue

        team = 'red' if agent_name.lower() == 'red' else 'blue'

        # Find starting host from starting_sessions
        starting_host = None
        starting_sessions = agent_info.get('starting_sessions', [])
        if starting_sessions:
            if team == 'blue':
                # Blue agent: find the server session (no parent, or VelociraptorServer type)
                for session in starting_sessions:
                    if 'parent' not in session or session.get('type') == 'VelociraptorServer':
                        starting_host = session.get('hostname')
                        break
                if starting_host is None:
                    starting_host = starting_sessions[0].get('hostname')
            else:
                # Red agent: use first session
                starting_host = starting_sessions[0].get('hostname')

        agent_configs.append(AgentConfig(
            name=agent_name,
            starting_host=starting_host,
            team=team,
        ))

    return ScenarioConfig(
        name=scenario_name,
        hosts=host_configs,
        subnets=subnet_configs,
        agents=agent_configs,
        max_steps=100,
    )


@lru_cache(maxsize=4)
def get_scenario_from_cyborg(scenario_name: str = 'Scenario2') -> ScenarioConfig:
    """Get a scenario config, loading from CybORG YAML files.

    This is the main entry point for loading scenarios. It attempts to load
    from CybORG files and raises an error if CybORG is not available.
    Results are cached for fast subsequent access.

    Args:
        scenario_name: Name of the scenario to load

    Returns:
        ScenarioConfig loaded from CybORG
    """
    return load_scenario_from_cyborg(scenario_name)
