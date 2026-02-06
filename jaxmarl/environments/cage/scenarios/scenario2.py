"""CAGE Challenge 2 - Scenario 2 configuration.

This module exports the scenario constants used by the CAGE-JAX environment.
The scenario is based on the original CAGE Challenge 2 specification.
"""

from jaxmarl.environments.cage.state import (
    create_scenario2_const,
    HOST_IDS,
    HOST_NAMES,
    SUBNET_IDS,
    SERVICE_IDS,
    EXPLOIT_IDS,
    DECOY_IDS,
    HOST_SUBNET,
)

# Scenario 2 constants
CONST = create_scenario2_const()

# Network topology description
SCENARIO_DESCRIPTION = """
CAGE Challenge 2 - Scenario 2

Network Structure:
- User Subnet (5 hosts): User0-4
- Enterprise Subnet (4 hosts): Enterprise0-2, Defender
- Operational Subnet (4 hosts): Op_Host0-2, Op_Server0

Connectivity:
- User <-> Enterprise
- Enterprise <-> Operational
- User cannot directly reach Operational

Red Agent:
- Starts with foothold on User0
- Goal: Reach and impact Op_Server0

Blue Agent:
- Defends all hosts
- Can remove malware, restore hosts, deploy decoys
"""

# Host configuration summary
HOST_CONFIG = {
    'User0': {'os': 'Linux', 'subnet': 'User', 'services': ['ssh'], 'value': 0.0},
    'User1': {'os': 'Linux', 'subnet': 'User', 'services': ['ssh'], 'value': 0.0},
    'User2': {'os': 'Linux', 'subnet': 'User', 'services': ['ssh'], 'value': 0.0},
    'User3': {'os': 'Linux', 'subnet': 'User', 'services': ['ssh'], 'value': 0.0},
    'User4': {'os': 'Linux', 'subnet': 'User', 'services': ['ssh'], 'value': 0.0},
    'Enterprise0': {'os': 'Windows', 'subnet': 'Enterprise', 'services': ['ssh', 'http'], 'value': 1.0},
    'Enterprise1': {'os': 'Windows', 'subnet': 'Enterprise', 'services': ['ssh', 'smb'], 'value': 1.0},
    'Enterprise2': {'os': 'Windows', 'subnet': 'Enterprise', 'services': ['ssh', 'tomcat'], 'value': 1.0},
    'Defender': {'os': 'Linux', 'subnet': 'Enterprise', 'services': ['ssh'], 'value': 0.0},
    'Op_Host0': {'os': 'Linux', 'subnet': 'Operational', 'services': ['ssh'], 'value': 1.0},
    'Op_Host1': {'os': 'Linux', 'subnet': 'Operational', 'services': ['ssh'], 'value': 1.0},
    'Op_Host2': {'os': 'Windows', 'subnet': 'Operational', 'services': ['rdp'], 'value': 1.0},
    'Op_Server0': {'os': 'Windows', 'subnet': 'Operational', 'services': ['ssh', 'http'], 'value': 10.0},
}


def get_scenario_const():
    """Return the Scenario 2 constants."""
    return CONST


def print_scenario_info():
    """Print scenario information."""
    print(SCENARIO_DESCRIPTION)
    print("\nHost Configuration:")
    for host, config in HOST_CONFIG.items():
        print(f"  {host}: {config}")


if __name__ == "__main__":
    print_scenario_info()
