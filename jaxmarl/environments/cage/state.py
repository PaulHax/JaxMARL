"""CAGE-JAX state dataclasses for Scenario 2."""

import jax.numpy as jnp
from flax import struct
import chex

NUM_HOSTS = 13
NUM_SUBNETS = 3
NUM_SERVICES = 10
NUM_EXPLOITS = 8
NUM_DECOY_TYPES = 8
MAX_PROCESSES = 20

# Host IDs (Scenario 2)
HOST_IDS = {
    'User0': 0, 'User1': 1, 'User2': 2, 'User3': 3, 'User4': 4,
    'Enterprise0': 5, 'Enterprise1': 6, 'Enterprise2': 7, 'Defender': 8,
    'Op_Host0': 9, 'Op_Host1': 10, 'Op_Host2': 11, 'Op_Server0': 12
}

# Reverse lookup
HOST_NAMES = {v: k for k, v in HOST_IDS.items()}

# Subnet IDs
SUBNET_IDS = {'User': 0, 'Enterprise': 1, 'Operational': 2}

# Host to subnet mapping
HOST_SUBNET = jnp.array([
    0, 0, 0, 0, 0,  # User0-4 -> User subnet
    1, 1, 1, 1,      # Enterprise0-2, Defender -> Enterprise subnet
    2, 2, 2, 2       # Op_Host0-2, Op_Server0 -> Operational subnet
], dtype=jnp.int32)

# Service IDs
SERVICE_IDS = {
    'ssh': 0, 'ftp': 1, 'http': 2, 'https': 3, 'smtp': 4,
    'mysql': 5, 'smb': 6, 'rdp': 7, 'tomcat': 8, 'haraka': 9
}

# Exploit IDs
EXPLOIT_IDS = {
    'SSHBruteForce': 0, 'FTPDirectoryTraversal': 1, 'HTTPRFI': 2,
    'HTTPSRFI': 3, 'HarakaRCE': 4, 'SQLInjection': 5,
    'EternalBlue': 6, 'BlueKeep': 7
}

# Decoy types (same as services for simplicity)
DECOY_IDS = {
    'DecoyApache': 0, 'DecoyTomcat': 1, 'DecoySSHD': 2, 'DecoyFemitter': 3,
    'DecoyHarakaSMPT': 4, 'DecoySvchost': 5, 'DecoySmss': 6, 'DecoyVsftpd': 7
}

# Compromise levels
COMPROMISE_NONE = 0
COMPROMISE_USER = 1
COMPROMISE_PRIVILEGED = 2

# OS types
OS_LINUX = 0
OS_WINDOWS = 1


@struct.dataclass
class CageState:
    """Mutable game state for CAGE environment."""
    time: int
    done: chex.Array  # scalar bool

    # Host state tensors
    host_compromised: chex.Array       # (13,) int: 0=None, 1=User, 2=Privileged
    host_services: chex.Array          # (13, 10) bool: service running flags
    host_processes: chex.Array         # (13, 20) int: process IDs (0=empty)
    host_decoys: chex.Array            # (13, 8) bool: decoy type deployed

    # Session tracking
    red_sessions: chex.Array           # (13,) int: session count per host
    red_privilege: chex.Array          # (13,) int: 0=None, 1=User, 2=Root/SYSTEM
    blue_sessions: chex.Array          # (13,) int: always 1 per host (Velociraptor)

    # Red agent knowledge (partial observability)
    red_discovered_hosts: chex.Array   # (13,) bool: host IP discovered
    red_scanned_hosts: chex.Array      # (13,) bool: ports scanned

    # Reward tracking
    cumulative_red_reward: chex.Array   # scalar float
    cumulative_blue_reward: chex.Array  # scalar float

    # Last action success (for observations)
    last_red_action_success: chex.Array  # scalar bool


@struct.dataclass
class CageConst:
    """Immutable configuration for CAGE environment (Scenario 2)."""
    # Network topology
    adjacency: chex.Array              # (13, 13) bool: can host i reach host j
    subnet_adjacency: chex.Array       # (3, 3) bool: subnet connectivity

    # Host properties
    host_os: chex.Array                # (13,) int: 0=Linux, 1=Windows
    host_confidentiality: chex.Array   # (13,) float: reward weight
    host_availability: chex.Array      # (13,) float: availability weight
    initial_services: chex.Array       # (13, 10) bool: initial service config

    # Service vulnerability mapping: which exploits work on which services
    service_exploits: chex.Array       # (10, 8) bool: service i vulnerable to exploit j

    # Scenario parameters
    max_steps: int = 100
    num_hosts: int = NUM_HOSTS
    num_subnets: int = NUM_SUBNETS


def create_scenario2_const() -> CageConst:
    """Create constants for CAGE Challenge 2 Scenario 2."""

    # Network adjacency - hosts can reach others in same or adjacent subnets
    # User subnet (0-4) <-> Enterprise subnet (5-8) <-> Operational subnet (9-12)
    adjacency = jnp.zeros((NUM_HOSTS, NUM_HOSTS), dtype=jnp.bool_)

    # Subnet adjacency: User <-> Enterprise <-> Operational
    subnet_adjacency = jnp.array([
        [True, True, False],   # User can reach Enterprise
        [True, True, True],    # Enterprise can reach both
        [False, True, True],   # Operational can reach Enterprise
    ], dtype=jnp.bool_)

    # Build host adjacency from subnet adjacency
    for i in range(NUM_HOSTS):
        for j in range(NUM_HOSTS):
            subnet_i = HOST_SUBNET[i]
            subnet_j = HOST_SUBNET[j]
            adjacency = adjacency.at[i, j].set(subnet_adjacency[subnet_i, subnet_j])

    # Host OS: User hosts are mixed, Enterprise/Operational have Windows servers
    # Simplified: User0-4 Linux, Enterprise0-2 Windows, Defender Linux, Op mixed
    host_os = jnp.array([
        OS_LINUX, OS_LINUX, OS_LINUX, OS_LINUX, OS_LINUX,  # User0-4
        OS_WINDOWS, OS_WINDOWS, OS_WINDOWS, OS_LINUX,       # Enterprise0-2, Defender
        OS_LINUX, OS_LINUX, OS_WINDOWS, OS_WINDOWS          # Op_Host0-2, Op_Server0
    ], dtype=jnp.int32)

    # Confidentiality values (higher = more valuable to compromise)
    host_confidentiality = jnp.array([
        0.0, 0.0, 0.0, 0.0, 0.0,      # User0-4: None
        1.0, 1.0, 1.0, 0.0,            # Enterprise0-2: Medium, Defender: None
        1.0, 1.0, 1.0, 10.0            # Op_Host0-2: Medium, Op_Server0: High
    ], dtype=jnp.float32)

    # Availability values
    host_availability = jnp.array([
        0.0, 0.0, 0.0, 0.0, 0.0,      # User0-4: None
        1.0, 1.0, 1.0, 0.0,            # Enterprise: Medium
        1.0, 1.0, 1.0, 10.0            # Operational: High for Op_Server0
    ], dtype=jnp.float32)

    # Initial services running on each host
    # Rows: hosts, Cols: services (ssh, ftp, http, https, smtp, mysql, smb, rdp, tomcat, haraka)
    initial_services = jnp.zeros((NUM_HOSTS, NUM_SERVICES), dtype=jnp.bool_)

    # User hosts: SSH
    for i in range(5):
        initial_services = initial_services.at[i, SERVICE_IDS['ssh']].set(True)

    # Enterprise hosts: various services
    initial_services = initial_services.at[HOST_IDS['Enterprise0'], SERVICE_IDS['ssh']].set(True)
    initial_services = initial_services.at[HOST_IDS['Enterprise0'], SERVICE_IDS['http']].set(True)
    initial_services = initial_services.at[HOST_IDS['Enterprise1'], SERVICE_IDS['ssh']].set(True)
    initial_services = initial_services.at[HOST_IDS['Enterprise1'], SERVICE_IDS['smb']].set(True)
    initial_services = initial_services.at[HOST_IDS['Enterprise2'], SERVICE_IDS['ssh']].set(True)
    initial_services = initial_services.at[HOST_IDS['Enterprise2'], SERVICE_IDS['tomcat']].set(True)
    initial_services = initial_services.at[HOST_IDS['Defender'], SERVICE_IDS['ssh']].set(True)

    # Operational hosts
    initial_services = initial_services.at[HOST_IDS['Op_Host0'], SERVICE_IDS['ssh']].set(True)
    initial_services = initial_services.at[HOST_IDS['Op_Host1'], SERVICE_IDS['ssh']].set(True)
    initial_services = initial_services.at[HOST_IDS['Op_Host2'], SERVICE_IDS['rdp']].set(True)
    initial_services = initial_services.at[HOST_IDS['Op_Server0'], SERVICE_IDS['ssh']].set(True)
    initial_services = initial_services.at[HOST_IDS['Op_Server0'], SERVICE_IDS['http']].set(True)

    # Service-exploit vulnerability mapping
    # Rows: services, Cols: exploits
    service_exploits = jnp.zeros((NUM_SERVICES, NUM_EXPLOITS), dtype=jnp.bool_)
    service_exploits = service_exploits.at[SERVICE_IDS['ssh'], EXPLOIT_IDS['SSHBruteForce']].set(True)
    service_exploits = service_exploits.at[SERVICE_IDS['ftp'], EXPLOIT_IDS['FTPDirectoryTraversal']].set(True)
    service_exploits = service_exploits.at[SERVICE_IDS['http'], EXPLOIT_IDS['HTTPRFI']].set(True)
    service_exploits = service_exploits.at[SERVICE_IDS['https'], EXPLOIT_IDS['HTTPSRFI']].set(True)
    service_exploits = service_exploits.at[SERVICE_IDS['haraka'], EXPLOIT_IDS['HarakaRCE']].set(True)
    service_exploits = service_exploits.at[SERVICE_IDS['mysql'], EXPLOIT_IDS['SQLInjection']].set(True)
    service_exploits = service_exploits.at[SERVICE_IDS['smb'], EXPLOIT_IDS['EternalBlue']].set(True)
    service_exploits = service_exploits.at[SERVICE_IDS['rdp'], EXPLOIT_IDS['BlueKeep']].set(True)

    return CageConst(
        adjacency=adjacency,
        subnet_adjacency=subnet_adjacency,
        host_os=host_os,
        host_confidentiality=host_confidentiality,
        host_availability=host_availability,
        initial_services=initial_services,
        service_exploits=service_exploits,
        max_steps=100,
        num_hosts=NUM_HOSTS,
        num_subnets=NUM_SUBNETS,
    )


def create_initial_state(const: CageConst) -> CageState:
    """Create initial state for CAGE environment."""
    return CageState(
        time=0,
        done=jnp.array(False),
        host_compromised=jnp.zeros(NUM_HOSTS, dtype=jnp.int32),
        host_services=const.initial_services.copy(),
        host_processes=jnp.zeros((NUM_HOSTS, MAX_PROCESSES), dtype=jnp.int32),
        host_decoys=jnp.zeros((NUM_HOSTS, NUM_DECOY_TYPES), dtype=jnp.bool_),
        red_sessions=jnp.zeros(NUM_HOSTS, dtype=jnp.int32),
        red_privilege=jnp.zeros(NUM_HOSTS, dtype=jnp.int32),
        blue_sessions=jnp.ones(NUM_HOSTS, dtype=jnp.int32),  # Blue has session on all hosts
        red_discovered_hosts=jnp.zeros(NUM_HOSTS, dtype=jnp.bool_),
        red_scanned_hosts=jnp.zeros(NUM_HOSTS, dtype=jnp.bool_),
        cumulative_red_reward=jnp.array(0.0),
        cumulative_blue_reward=jnp.array(0.0),
        last_red_action_success=jnp.array(False),
    )


def create_initial_state_with_red_foothold(const: CageConst) -> CageState:
    """Create initial state where Red has a foothold on User0 (standard CC2 setup)."""
    state = create_initial_state(const)

    # Red starts with User-level access on User0
    state = state.replace(
        host_compromised=state.host_compromised.at[HOST_IDS['User0']].set(COMPROMISE_USER),
        red_sessions=state.red_sessions.at[HOST_IDS['User0']].set(1),
        red_privilege=state.red_privilege.at[HOST_IDS['User0']].set(COMPROMISE_USER),
        red_discovered_hosts=state.red_discovered_hosts.at[HOST_IDS['User0']].set(True),
        red_scanned_hosts=state.red_scanned_hosts.at[HOST_IDS['User0']].set(True),
    )

    return state
