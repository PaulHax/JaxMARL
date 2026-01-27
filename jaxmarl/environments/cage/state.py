"""CAGE-JAX state dataclasses with configurable scenario support."""

import jax.numpy as jnp
from flax import struct
import chex
from typing import Dict, Optional

from jaxmarl.environments.cage.config import ScenarioConfig, create_scenario2_config


# Default constants for backward compatibility (Scenario 2)
NUM_HOSTS = 13
NUM_SUBNETS = 3
NUM_SERVICES = 10
NUM_EXPLOITS = 8
NUM_DECOY_TYPES = 8
MAX_PROCESSES = 20

# Host IDs (Scenario 2) - alphabetical order to match CybORG
HOST_IDS = {
    'Defender': 0, 'Enterprise0': 1, 'Enterprise1': 2, 'Enterprise2': 3,
    'Op_Host0': 4, 'Op_Host1': 5, 'Op_Host2': 6, 'Op_Server0': 7,
    'User0': 8, 'User1': 9, 'User2': 10, 'User3': 11, 'User4': 12,
}

# Reverse lookup
HOST_NAMES = {v: k for k, v in HOST_IDS.items()}

# Subnet IDs
SUBNET_IDS = {'User': 0, 'Enterprise': 1, 'Operational': 2}

# Host to subnet mapping (matching alphabetical host order)
HOST_SUBNET = jnp.array([
    1,              # Defender -> Enterprise subnet
    1, 1, 1,        # Enterprise0-2 -> Enterprise subnet
    2, 2, 2, 2,     # Op_Host0-2, Op_Server0 -> Operational subnet
    0, 0, 0, 0, 0,  # User0-4 -> User subnet
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

# Decoy types (alphabetical order to match CybORG)
DECOY_IDS = {
    'DecoyApache': 0, 'DecoyFemitter': 1, 'DecoyHarakaSMPT': 2, 'DecoySmss': 3,
    'DecoySSHD': 4, 'DecoySvchost': 5, 'DecoyTomcat': 6, 'DecoyVsftpd': 7,
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
    """Mutable game state for CAGE environment.

    All arrays are sized based on max dimensions to support JIT compilation.
    Actual sizes are determined by const.num_hosts, const.num_subnets, etc.
    """
    time: int
    done: chex.Array  # scalar bool

    # Host state tensors
    host_compromised: chex.Array       # (max_hosts,) int: 0=None, 1=User, 2=Privileged
    host_services: chex.Array          # (max_hosts, max_services) bool: service running flags
    host_processes: chex.Array         # (max_hosts, max_processes) int: process IDs (0=empty)
    host_decoys: chex.Array            # (max_hosts, max_decoys) bool: decoy type deployed

    # Session tracking
    red_sessions: chex.Array           # (max_hosts,) int: session count per host
    red_privilege: chex.Array          # (max_hosts,) int: 0=None, 1=User, 2=Root/SYSTEM
    blue_sessions: chex.Array          # (max_hosts,) int: always 1 per host (Velociraptor)

    # Red agent knowledge (partial observability)
    # NOTE: Fields with _jax suffix are JAX-specific simplifications for action masking.
    # CybORG doesn't track discovered/scanned hosts explicitly - it checks
    # network routing dynamically at action time. JAX uses explicit state to
    # avoid recomputing routing on every action, which is more efficient for
    # vectorized environments. The game outcomes (rewards, compromise) match
    # CybORG, but these internal tracking fields will differ in comparisons.
    red_discovered_hosts_jax: chex.Array   # (max_hosts,) bool: host IP discovered
    red_scanned_hosts_jax: chex.Array      # (max_hosts,) bool: ports scanned

    # Reward tracking
    cumulative_red_reward: chex.Array   # scalar float
    cumulative_blue_reward: chex.Array  # scalar float

    # Last action success (for observations)
    last_red_action_success: chex.Array  # scalar bool

    # Impact tracking - OT service stopped on operational hosts
    ot_service_stopped: chex.Array     # (max_hosts,) bool: OT service stopped by Impact

    # Activity detection - tracks if suspicious activity was detected (for Remove)
    host_activity_detected: chex.Array  # (max_hosts,) bool: activity detected by Monitor/Analyse

    # Unknown observation state - set after Remove, cleared by Monitor/Analyse
    host_observation_unknown: chex.Array  # (max_hosts,) bool: observation should show "Unknown"

    # Red activity this step - tracks hosts where Red took action this step (for Monitor detection)
    # Cleared at start of each step, set by Red actions. Monitor detects this, not pre-existing state.
    red_activity_this_step: chex.Array  # (max_hosts,) bool: Red took action on host this step


@struct.dataclass
class CageConst:
    """Immutable configuration for CAGE environment."""
    # Network topology
    adjacency: chex.Array              # (num_hosts, num_hosts) bool: can host i reach host j
    subnet_adjacency: chex.Array       # (num_subnets, num_subnets) bool: subnet connectivity
    host_subnet: chex.Array            # (num_hosts,) int: which subnet each host belongs to

    # Host properties
    host_os: chex.Array                # (num_hosts,) int: 0=Linux, 1=Windows
    host_confidentiality: chex.Array   # (num_hosts,) float: reward weight
    host_availability: chex.Array      # (num_hosts,) float: availability weight
    initial_services: chex.Array       # (num_hosts, num_services) bool: initial service config
    operational_targets: chex.Array    # (num_hosts,) bool: hosts that provide availability reward

    # Service vulnerability mapping: which exploits work on which services
    service_exploits: chex.Array       # (num_services, num_exploits) bool: service i vulnerable to exploit j

    # Indices for decoy-deployable hosts
    decoy_host_indices: chex.Array     # (num_decoy_hosts,) int: indices of hosts that can have decoys

    # Red agent initial foothold
    red_start_hosts: chex.Array        # (num_red_agents,) int: initial compromise host indices

    # B_lineAgent target host indices (looked up by name for scenario compatibility)
    bline_user_host: int = 9           # User1 - initial target in User subnet
    bline_enterprise0: int = 1         # Enterprise0
    bline_enterprise2: int = 3         # Enterprise2
    bline_op_server0: int = 7          # Op_Server0 - final target for Impact

    # Scenario parameters
    max_steps: int = 100
    num_hosts: int = NUM_HOSTS
    num_subnets: int = NUM_SUBNETS
    num_services: int = NUM_SERVICES
    num_exploits: int = NUM_EXPLOITS
    num_decoys: int = NUM_DECOY_TYPES
    num_decoy_hosts: int = 13  # All hosts can have decoys (matching CybORG)
    num_red_agents: int = 1
    num_blue_agents: int = 1


def build_const_from_config(config: ScenarioConfig) -> CageConst:
    """Build CageConst from ScenarioConfig."""
    num_hosts = config.num_hosts
    num_subnets = config.num_subnets
    num_services = config.num_services
    num_exploits = config.num_exploits
    num_decoys = config.num_decoys

    host_ids = config.host_ids
    subnet_ids = config.subnet_ids
    service_ids = config.service_ids
    exploit_ids = config.exploit_ids

    # Build host_subnet mapping
    host_subnet = jnp.zeros(num_hosts, dtype=jnp.int32)
    for h in config.hosts:
        host_subnet = host_subnet.at[host_ids[h.name]].set(subnet_ids[h.subnet])

    # Build subnet adjacency from config
    subnet_adjacency = jnp.zeros((num_subnets, num_subnets), dtype=jnp.bool_)
    for s in config.subnets:
        src_idx = subnet_ids[s.name]
        for connected in s.connected_subnets:
            dst_idx = subnet_ids[connected]
            subnet_adjacency = subnet_adjacency.at[src_idx, dst_idx].set(True)

    # Build host adjacency from subnet adjacency
    adjacency = jnp.zeros((num_hosts, num_hosts), dtype=jnp.bool_)
    for i in range(num_hosts):
        for j in range(num_hosts):
            subnet_i = int(host_subnet[i])
            subnet_j = int(host_subnet[j])
            adjacency = adjacency.at[i, j].set(subnet_adjacency[subnet_i, subnet_j])

    # Build host OS array
    host_os = jnp.zeros(num_hosts, dtype=jnp.int32)
    for h in config.hosts:
        os_val = OS_WINDOWS if h.os.lower() == 'windows' else OS_LINUX
        host_os = host_os.at[host_ids[h.name]].set(os_val)

    # Build confidentiality and availability arrays
    host_confidentiality = jnp.zeros(num_hosts, dtype=jnp.float32)
    host_availability = jnp.zeros(num_hosts, dtype=jnp.float32)
    operational_targets = jnp.zeros(num_hosts, dtype=jnp.bool_)

    for h in config.hosts:
        idx = host_ids[h.name]
        host_confidentiality = host_confidentiality.at[idx].set(h.confidentiality)
        host_availability = host_availability.at[idx].set(h.availability)
        if h.is_operational_target:
            operational_targets = operational_targets.at[idx].set(True)

    # Build initial services
    initial_services = jnp.zeros((num_hosts, num_services), dtype=jnp.bool_)
    for h in config.hosts:
        host_idx = host_ids[h.name]
        for svc in h.services:
            if svc in service_ids:
                svc_idx = service_ids[svc]
                initial_services = initial_services.at[host_idx, svc_idx].set(True)

    # Build service-exploit vulnerability matrix
    service_exploits = jnp.zeros((num_services, num_exploits), dtype=jnp.bool_)
    for svc, vulns in config.service_vulnerabilities.items():
        if svc in service_ids:
            svc_idx = service_ids[svc]
            for exploit in vulns:
                if exploit in exploit_ids:
                    exp_idx = exploit_ids[exploit]
                    service_exploits = service_exploits.at[svc_idx, exp_idx].set(True)

    # Build decoy host indices
    decoy_indices = config.decoy_host_indices
    decoy_host_indices = jnp.array(decoy_indices, dtype=jnp.int32)

    # Build red start hosts
    red_agents = config.get_red_agents()
    red_start_hosts = jnp.zeros(len(red_agents), dtype=jnp.int32)
    for i, agent in enumerate(red_agents):
        if agent.starting_host:
            red_start_hosts = red_start_hosts.at[i].set(host_ids[agent.starting_host])

    # Look up B_lineAgent target hosts by name (works for any scenario)
    bline_user_host = host_ids.get('User1', 9)
    bline_enterprise0 = host_ids.get('Enterprise0', 1)
    bline_enterprise2 = host_ids.get('Enterprise2', 3)
    bline_op_server0 = host_ids.get('Op_Server0', 7)

    return CageConst(
        adjacency=adjacency,
        subnet_adjacency=subnet_adjacency,
        host_subnet=host_subnet,
        host_os=host_os,
        host_confidentiality=host_confidentiality,
        host_availability=host_availability,
        initial_services=initial_services,
        operational_targets=operational_targets,
        service_exploits=service_exploits,
        decoy_host_indices=decoy_host_indices,
        red_start_hosts=red_start_hosts,
        bline_user_host=bline_user_host,
        bline_enterprise0=bline_enterprise0,
        bline_enterprise2=bline_enterprise2,
        bline_op_server0=bline_op_server0,
        max_steps=config.max_steps,
        num_hosts=num_hosts,
        num_subnets=num_subnets,
        num_services=num_services,
        num_exploits=num_exploits,
        num_decoys=num_decoys,
        num_decoy_hosts=len(decoy_indices),
        num_red_agents=config.num_red_agents,
        num_blue_agents=config.num_blue_agents,
    )


def create_scenario2_const() -> CageConst:
    """Create constants for CAGE Challenge 2 Scenario 2."""
    config = create_scenario2_config()
    return build_const_from_config(config)


def create_const_from_scenario(scenario_name: str) -> CageConst:
    """Create CageConst from a named scenario."""
    from jaxmarl.environments.cage.config import get_scenario
    config = get_scenario(scenario_name)
    return build_const_from_config(config)


def create_initial_state(const: CageConst) -> CageState:
    """Create initial state for CAGE environment."""
    num_hosts = const.num_hosts
    num_services = const.num_services
    num_decoys = const.num_decoys

    return CageState(
        time=0,
        done=jnp.array(False),
        host_compromised=jnp.zeros(num_hosts, dtype=jnp.int32),
        host_services=const.initial_services.copy(),
        host_processes=jnp.zeros((num_hosts, MAX_PROCESSES), dtype=jnp.int32),
        host_decoys=jnp.zeros((num_hosts, num_decoys), dtype=jnp.bool_),
        red_sessions=jnp.zeros(num_hosts, dtype=jnp.int32),
        red_privilege=jnp.zeros(num_hosts, dtype=jnp.int32),
        blue_sessions=jnp.ones(num_hosts, dtype=jnp.int32),
        red_discovered_hosts_jax=jnp.zeros(num_hosts, dtype=jnp.bool_),
        red_scanned_hosts_jax=jnp.zeros(num_hosts, dtype=jnp.bool_),
        cumulative_red_reward=jnp.array(0.0),
        cumulative_blue_reward=jnp.array(0.0),
        last_red_action_success=jnp.array(False),
        ot_service_stopped=jnp.zeros(num_hosts, dtype=jnp.bool_),
        host_activity_detected=jnp.zeros(num_hosts, dtype=jnp.bool_),
        host_observation_unknown=jnp.zeros(num_hosts, dtype=jnp.bool_),
        red_activity_this_step=jnp.zeros(num_hosts, dtype=jnp.bool_),
    )


def create_initial_state_with_red_foothold(const: CageConst) -> CageState:
    """Create initial state where Red has a foothold on configured start hosts.

    CybORG starts Red with SYSTEM/root (PRIVILEGED) access on the foothold host.
    """
    state = create_initial_state(const)

    # Set up red foothold using scatter operations (JAX-compatible)
    red_start_mask = jnp.zeros(const.num_hosts, dtype=jnp.bool_)
    red_start_mask = red_start_mask.at[const.red_start_hosts].set(True)

    state = state.replace(
        host_compromised=jnp.where(red_start_mask, COMPROMISE_PRIVILEGED, state.host_compromised),
        red_sessions=jnp.where(red_start_mask, 1, state.red_sessions),
        red_privilege=jnp.where(red_start_mask, COMPROMISE_PRIVILEGED, state.red_privilege),
        red_discovered_hosts_jax=red_start_mask | state.red_discovered_hosts_jax,
        red_scanned_hosts_jax=red_start_mask | state.red_scanned_hosts_jax,
    )

    return state
