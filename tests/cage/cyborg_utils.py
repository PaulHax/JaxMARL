"""CybORG utility functions for equivalence testing.

This module provides helpers for:
- Creating CybORG environments
- Extracting comparable state from CybORG
- Translating actions between CAGE-JAX and CybORG
- Comparing states between the two implementations
"""

from pathlib import Path
from typing import Optional
import importlib.util

import jax.numpy as jnp

from jaxmarl.environments.cage.state import (
    HOST_IDS, HOST_NAMES, NUM_HOSTS,
    COMPROMISE_NONE, COMPROMISE_USER, COMPROMISE_PRIVILEGED,
    CageState,
)
from jaxmarl.environments.cage.actions import (
    BLUE_SLEEP, BLUE_MONITOR, BLUE_REMOVE_START, BLUE_RESTORE_START, BLUE_DECOY_START,
    RED_SLEEP, RED_DISCOVER_SUBNET_START, RED_SCAN_HOST_START, RED_EXPLOIT_START,
    RED_PRIVESC_START, RED_IMPACT_START,
    NUM_HOSTS as ACTION_NUM_HOSTS,
)


def get_cyborg_scenario_path(scenario_name: str = "Scenario2.yaml") -> Path:
    """Get path to CybORG scenario file using importlib."""
    spec = importlib.util.find_spec("CybORG")
    if spec is None or spec.origin is None:
        raise ImportError("CybORG package not found")
    cyborg_path = Path(spec.origin).parent
    scenario_path = cyborg_path / "Shared" / "Scenarios" / scenario_name
    if not scenario_path.exists():
        raise FileNotFoundError(f"Scenario file not found: {scenario_path}")
    return scenario_path


def create_cyborg_env(scenario: str = "Scenario2.yaml", seed: Optional[int] = None):
    """Create a CybORG environment instance.

    Args:
        scenario: Scenario file name
        seed: Optional random seed

    Returns:
        CybORG instance
    """
    from CybORG import CybORG

    scenario_path = get_cyborg_scenario_path(scenario)
    env = CybORG(scenario_file=str(scenario_path), environment='sim')
    if seed is not None:
        env.set_seed(seed)
    return env


def cyborg_state_to_dict(cyborg_env) -> dict:
    """Extract comparable state from CybORG environment.

    Returns a dictionary with:
    - host_compromised: {hostname: compromise_level} for each host
    - red_sessions: {hostname: session_count}
    - red_privilege: {hostname: privilege_level}
    """
    host_compromised = {}
    red_sessions = {}
    red_privilege = {}

    for hostname in HOST_IDS.keys():
        host_compromised[hostname] = COMPROMISE_NONE
        red_sessions[hostname] = 0
        red_privilege[hostname] = COMPROMISE_NONE

    red_state = cyborg_env.get_agent_state('Red')

    for hostname in HOST_IDS.keys():
        if hostname in red_state:
            host_info = red_state[hostname]
            sessions = host_info.get('Sessions', [])
            for session_info in sessions:
                if isinstance(session_info, dict):
                    agent = session_info.get('Agent', session_info.get('agent', ''))
                    if agent == 'Red':
                        red_sessions[hostname] = red_sessions.get(hostname, 0) + 1
                        username = session_info.get('Username', session_info.get('username', ''))

                        if username in ['root', 'SYSTEM']:
                            red_privilege[hostname] = COMPROMISE_PRIVILEGED
                            host_compromised[hostname] = COMPROMISE_PRIVILEGED
                        elif red_privilege[hostname] < COMPROMISE_USER:
                            red_privilege[hostname] = COMPROMISE_USER
                            if host_compromised[hostname] < COMPROMISE_USER:
                                host_compromised[hostname] = COMPROMISE_USER

    return {
        'host_compromised': host_compromised,
        'red_sessions': red_sessions,
        'red_privilege': red_privilege,
    }


def jax_state_to_dict(state: CageState) -> dict:
    """Extract comparable state from CAGE-JAX CageState.

    Returns a dictionary with same structure as cyborg_state_to_dict.
    """
    host_compromised = {}
    red_sessions = {}
    red_privilege = {}

    for hostname, idx in HOST_IDS.items():
        host_compromised[hostname] = int(state.host_compromised[idx])
        red_sessions[hostname] = int(state.red_sessions[idx])
        red_privilege[hostname] = int(state.red_privilege[idx])

    return {
        'host_compromised': host_compromised,
        'red_sessions': red_sessions,
        'red_privilege': red_privilege,
    }


def states_match(cyborg_state: dict, jax_state: dict, tolerance: float = 0.0) -> bool:
    """Compare CybORG and CAGE-JAX states for equivalence.

    Args:
        cyborg_state: State dict from cyborg_state_to_dict
        jax_state: State dict from jax_state_to_dict
        tolerance: Allowed tolerance for numerical comparisons

    Returns:
        True if states are equivalent
    """
    for hostname in HOST_IDS.keys():
        if cyborg_state['host_compromised'].get(hostname, 0) != jax_state['host_compromised'].get(hostname, 0):
            return False

        cyborg_priv = cyborg_state['red_privilege'].get(hostname, 0)
        jax_priv = jax_state['red_privilege'].get(hostname, 0)
        if cyborg_priv != jax_priv:
            return False

    return True


def get_state_diff(cyborg_state: dict, jax_state: dict) -> dict:
    """Get differences between CybORG and CAGE-JAX states.

    Returns:
        Dictionary mapping field names to (cyborg_value, jax_value) tuples
    """
    diffs = {}

    for hostname in HOST_IDS.keys():
        cyborg_comp = cyborg_state['host_compromised'].get(hostname, 0)
        jax_comp = jax_state['host_compromised'].get(hostname, 0)
        if cyborg_comp != jax_comp:
            diffs[f'{hostname}_compromised'] = (cyborg_comp, jax_comp)

        cyborg_priv = cyborg_state['red_privilege'].get(hostname, 0)
        jax_priv = jax_state['red_privilege'].get(hostname, 0)
        if cyborg_priv != jax_priv:
            diffs[f'{hostname}_privilege'] = (cyborg_priv, jax_priv)

        cyborg_sess = cyborg_state['red_sessions'].get(hostname, 0)
        jax_sess = jax_state['red_sessions'].get(hostname, 0)
        if cyborg_sess != jax_sess:
            diffs[f'{hostname}_sessions'] = (cyborg_sess, jax_sess)

    return diffs


def jax_blue_action_to_cyborg(action_idx: int, cyborg_env):
    """Convert CAGE-JAX blue action index to CybORG action.

    Args:
        action_idx: CAGE-JAX action index
        cyborg_env: CybORG environment for action space access

    Returns:
        CybORG action object
    """
    from CybORG.Shared.Actions import Sleep, Monitor, Analyse, Remove, Restore

    if action_idx == BLUE_SLEEP:
        return Sleep()
    elif action_idx == BLUE_MONITOR:
        return Monitor(session=0, agent='Blue')

    if BLUE_REMOVE_START <= action_idx < BLUE_RESTORE_START:
        host_idx = action_idx - BLUE_REMOVE_START
        hostname = HOST_NAMES[host_idx]
        return Remove(session=0, agent='Blue', hostname=hostname)

    if BLUE_RESTORE_START <= action_idx < BLUE_DECOY_START:
        host_idx = action_idx - BLUE_RESTORE_START
        hostname = HOST_NAMES[host_idx]
        return Restore(session=0, agent='Blue', hostname=hostname)

    return Sleep()


def jax_red_action_to_cyborg(action_idx: int, cyborg_env):
    """Convert CAGE-JAX red action index to CybORG action.

    Args:
        action_idx: CAGE-JAX action index
        cyborg_env: CybORG environment for action space access

    Returns:
        CybORG action object
    """
    from CybORG.Shared.Actions import (
        Sleep, DiscoverRemoteSystems, DiscoverNetworkServices,
        ExploitRemoteService, PrivilegeEscalate, Impact,
        SSHBruteForce, FTPDirectoryTraversal, HTTPRFI, HTTPSRFI,
        HarakaRCE, SQLInjection, EternalBlue, BlueKeep,
    )

    SUBNET_NAMES = {0: 'User', 1: 'Enterprise', 2: 'Operational'}
    EXPLOIT_CLASSES = [
        SSHBruteForce, FTPDirectoryTraversal, HTTPRFI, HTTPSRFI,
        HarakaRCE, SQLInjection, EternalBlue, BlueKeep,
    ]

    if action_idx == RED_SLEEP:
        return Sleep()

    if RED_DISCOVER_SUBNET_START <= action_idx < RED_SCAN_HOST_START:
        subnet_idx = action_idx - RED_DISCOVER_SUBNET_START
        subnet_name = SUBNET_NAMES.get(subnet_idx, 'User')
        return DiscoverRemoteSystems(session=0, agent='Red', subnet=subnet_name)

    if RED_SCAN_HOST_START <= action_idx < RED_EXPLOIT_START:
        host_idx = action_idx - RED_SCAN_HOST_START
        hostname = HOST_NAMES[host_idx]
        ip = cyborg_env.get_ip_map().get(hostname)
        return DiscoverNetworkServices(session=0, agent='Red', ip_address=ip)

    if RED_EXPLOIT_START <= action_idx < RED_PRIVESC_START:
        offset = action_idx - RED_EXPLOIT_START
        exploit_idx = offset // ACTION_NUM_HOSTS
        host_idx = offset % ACTION_NUM_HOSTS
        hostname = HOST_NAMES[host_idx]
        ip = cyborg_env.get_ip_map().get(hostname)

        if exploit_idx < len(EXPLOIT_CLASSES):
            exploit_class = EXPLOIT_CLASSES[exploit_idx]
            return exploit_class(ip_address=ip, agent='Red', session=0, target_session=0)
        return Sleep()

    if RED_PRIVESC_START <= action_idx < RED_IMPACT_START:
        host_idx = action_idx - RED_PRIVESC_START
        hostname = HOST_NAMES[host_idx]
        return PrivilegeEscalate(session=0, agent='Red', hostname=hostname)

    if action_idx >= RED_IMPACT_START:
        host_idx = action_idx - RED_IMPACT_START
        hostname = HOST_NAMES[host_idx]
        return Impact(session=0, agent='Red', hostname=hostname)

    return Sleep()


def run_cyborg_episode(cyborg_env, blue_actions: list, red_actions: list, max_steps: int = 100):
    """Run a CybORG episode with specified actions.

    Args:
        cyborg_env: CybORG environment
        blue_actions: List of CybORG blue actions
        red_actions: List of CybORG red actions
        max_steps: Maximum steps to run

    Returns:
        List of (state_dict, rewards) tuples for each step
    """
    from CybORG.Shared.Actions import Sleep

    cyborg_env.reset()
    trajectory = []

    for t in range(min(max_steps, len(blue_actions), len(red_actions))):
        blue_action = blue_actions[t] if t < len(blue_actions) else Sleep()
        red_action = red_actions[t] if t < len(red_actions) else Sleep()

        cyborg_env.step('Blue', blue_action)
        result = cyborg_env.step('Red', red_action)

        state = cyborg_state_to_dict(cyborg_env)
        rewards = cyborg_env.get_rewards()

        trajectory.append({
            'state': state,
            'rewards': rewards,
            'step': t,
        })

    return trajectory


def run_jax_episode(jax_env, blue_actions: list, red_actions: list, key, max_steps: int = 100):
    """Run a CAGE-JAX episode with specified actions.

    Args:
        jax_env: CageEnv instance
        blue_actions: List of CAGE-JAX blue action indices
        red_actions: List of CAGE-JAX red action indices
        key: JAX PRNG key
        max_steps: Maximum steps to run

    Returns:
        List of (state_dict, rewards) tuples for each step
    """
    import jax

    obs, state = jax_env.reset(key)
    trajectory = []

    for t in range(min(max_steps, len(blue_actions), len(red_actions))):
        key, subkey = jax.random.split(key)

        actions = {
            'blue': jnp.array(blue_actions[t]),
            'red': jnp.array(red_actions[t]),
        }

        obs, state, rewards, dones, info = jax_env.step_env(subkey, state, actions)

        state_dict = jax_state_to_dict(state)

        trajectory.append({
            'state': state_dict,
            'rewards': {k: float(v) for k, v in rewards.items()},
            'step': t,
        })

        if dones['__all__']:
            break

    return trajectory
