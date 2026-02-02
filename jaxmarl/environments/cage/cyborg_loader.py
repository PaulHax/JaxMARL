"""Load JaxMARL scenario configuration from CybORG YAML files.

This module provides functions to parse CybORG's YAML scenario and image files
and convert them to JaxMARL's ScenarioConfig format. This ensures JaxMARL
stays in sync with CybORG's scenario definitions without manual duplication.
"""

from pathlib import Path
from typing import Dict, List, Tuple
import yaml

from jaxmarl.environments.cage.config import (
    HostConfig, SubnetConfig, AgentConfig, ScenarioConfig
)


PORT_TO_SERVICE = {
    22: 'ssh',
    21: 'ftp',
    80: 'http',
    443: 'https',
    25: 'smtp',
    139: 'smb',
    445: 'smb',
}

PORT_PROCESS_TYPE_TO_SERVICE = {
    (3389, 'rdp'): 'rdp',
}

PROCESS_NAME_TO_SERVICE = {
    'mysql': 'mysql',
    'tomcat8.exe': 'tomcat',
}

CONFIDENTIALITY_MAP = {
    'None': 0.0,
    'Low': 0.1,
    'Medium': 1.0,
    'High': 10.0,
}

AVAILABILITY_MAP = {
    'None': 0.0,
    'Low': 0.1,
    'Medium': 1.0,
    'High': 10.0,
}


def load_yaml_file(file_path: Path) -> dict:
    """Load a YAML file and return its contents as a dictionary."""
    with open(file_path) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def extract_services_from_processes(processes: List[dict]) -> Tuple[List[str], Dict[str, List[str]]]:
    """Extract service names and their properties from CybORG process definitions.

    Services are identified by:
    1. Port number (e.g., port 22 → ssh)
    2. Process name (e.g., mysql → mysql)
    3. Process Type for services like RDP
    4. Process Version for haraka detection

    Returns:
        Tuple of (services list, service_properties dict)
        service_properties maps service name to list of properties (e.g., {'http': ['rfi']})
    """
    services = set()
    service_properties: Dict[str, List[str]] = {}

    for process in processes:
        connections = process.get('Connections', [])
        process_name = process.get('Process Name', '').lower()
        process_version = process.get('Process Version', '')
        process_type = process.get('Process Type', '').lower()
        raw_props = process.get('Properties', [])
        # Normalize properties: can be a list ['rfi'] or a string 'rfi'
        if isinstance(raw_props, str):
            properties = [raw_props]
        else:
            properties = list(raw_props) if raw_props else []

        # Check connections for port-based services
        for conn in connections:
            port = conn.get('local_port')
            if port in PORT_TO_SERVICE:
                service = PORT_TO_SERVICE[port]
                # Special case: port 25 with haraka version
                if port == 25 and 'haraka' in process_version.lower():
                    services.add('haraka')
                    if properties:
                        service_properties['haraka'] = list(properties)
                else:
                    services.add(service)
                    if properties:
                        service_properties[service] = list(properties)

            # Check port + process type combinations
            key = (port, process_type)
            if key in PORT_PROCESS_TYPE_TO_SERVICE:
                svc = PORT_PROCESS_TYPE_TO_SERVICE[key]
                services.add(svc)
                if properties:
                    service_properties[svc] = list(properties)

        # Check process name for additional services
        for proc_pattern, service in PROCESS_NAME_TO_SERVICE.items():
            if proc_pattern.lower() in process_name:
                services.add(service)
                if properties:
                    service_properties[service] = list(properties)

    return sorted(services), service_properties


def extract_os_type(system_info: dict) -> str:
    """Extract OS type from CybORG system info."""
    os_type = system_info.get('OSType', 'LINUX')
    return 'windows' if os_type.upper() == 'WINDOWS' else 'linux'


def extract_has_bruteforceable_users(user_info) -> bool:
    """Check if any user on the host has Bruteforceable=True.

    User info is a list of user dicts, each containing Username, Bruteforceable, etc.
    CybORG uses 'Bruteforceable' key (capital B).
    """
    if isinstance(user_info, list):
        for user_data in user_info:
            if isinstance(user_data, dict) and user_data.get('Bruteforceable', False):
                return True
    elif isinstance(user_info, dict):
        for username, user_data in user_info.items():
            if isinstance(user_data, dict) and user_data.get('Bruteforceable', False):
                return True
    return False


def load_scenario_from_yaml(
    scenario_path: Path,
    images_path: Path,
) -> ScenarioConfig:
    """Load scenario configuration from YAML files.

    Args:
        scenario_path: Path to the scenario YAML file.
        images_path: Path to the images directory containing images.yaml
                     and host image definitions.

    Returns:
        ScenarioConfig populated from the YAML files.

    Raises:
        FileNotFoundError: If scenario or images files cannot be found.
    """
    scenario_path = Path(scenario_path)
    images_path = Path(images_path)

    if not scenario_path.exists():
        raise FileNotFoundError(f"Scenario file not found: {scenario_path}")
    if not images_path.exists():
        raise FileNotFoundError(f"Images directory not found: {images_path}")

    scenario_data = load_yaml_file(scenario_path)
    scenario_name = scenario_path.stem

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

        # Extract services and their properties from processes
        processes = host_image_data.get('Processes', [])
        services, service_properties = extract_services_from_processes(processes)

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

        # Check for bruteforceable users (for SSHBruteForce)
        user_info = host_image_data.get('User Info', {})
        has_bruteforceable_users = extract_has_bruteforceable_users(user_info)

        host_configs.append(HostConfig(
            name=host_name,
            subnet=host_subnet,
            os=os_type,
            confidentiality=confidentiality,
            availability=availability,
            services=services,
            service_properties=service_properties,
            is_operational_target=is_operational_target,
            has_bruteforceable_users=has_bruteforceable_users,
        ))

    # Build subnet configs with connectivity
    # NACLs define bidirectional rules: source must allow 'out' AND target must allow 'in'
    all_subnet_names = list(subnets_data.keys())

    def can_reach(src_subnet: str, dst_subnet: str) -> bool:
        """Check if src_subnet can reach dst_subnet based on NACLs."""
        if src_subnet == dst_subnet:
            return True

        src_nacls = subnets_data[src_subnet].get('NACLs', {})
        dst_nacls = subnets_data[dst_subnet].get('NACLs', {})

        # Check source allows outgoing to destination
        src_allows_out = False
        if 'all' in src_nacls:
            out_rule = src_nacls['all'].get('out', 'None')
            if out_rule == 'all' or out_rule != 'None':
                src_allows_out = True
        if dst_subnet in src_nacls:
            out_rule = src_nacls[dst_subnet].get('out', 'None')
            if out_rule == 'all' or out_rule != 'None':
                src_allows_out = True

        # Check destination allows incoming from source
        dst_allows_in = False
        dst_blocks_in = False
        if 'all' in dst_nacls:
            in_rule = dst_nacls['all'].get('in', 'None')
            if in_rule == 'all' or in_rule != 'None':
                dst_allows_in = True
        if src_subnet in dst_nacls:
            in_rule = dst_nacls[src_subnet].get('in', 'None')
            if in_rule == 'None':
                dst_blocks_in = True
            elif in_rule == 'all' or in_rule != 'None':
                dst_allows_in = True

        return src_allows_out and dst_allows_in and not dst_blocks_in

    subnet_configs = []
    for subnet_name in all_subnet_names:
        subnet_info = subnets_data[subnet_name]
        hosts_in_subnet = [h.name for h in host_configs if h.subnet == subnet_name]

        connected = set()
        for other_subnet in all_subnet_names:
            if can_reach(subnet_name, other_subnet):
                connected.add(other_subnet)

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


