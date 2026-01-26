"""Comprehensive state extraction and comparison between CybORG and CAGE-JAX.

This module provides functions to extract comparable state from both
environments and detect discrepancies.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import jax.numpy as jnp

from jaxmarl.environments.cage.state import (
    HOST_IDS, HOST_NAMES, SUBNET_IDS,
    COMPROMISE_NONE, COMPROMISE_USER, COMPROMISE_PRIVILEGED,
    CageState, CageConst,
)
from jaxmarl.environments.cage.observations import get_blue_obs, get_red_obs


@dataclass
class StateDiff:
    """Represents a difference between CybORG and JAX state."""
    field: str
    host: Optional[str]
    cyborg_value: Any
    jax_value: Any
    severity: str = 'error'

    def __str__(self) -> str:
        host_str = f" ({self.host})" if self.host else ""
        return f"[{self.severity.upper()}] {self.field}{host_str}: CybORG={self.cyborg_value}, JAX={self.jax_value}"


@dataclass
class StateSnapshot:
    """Complete state snapshot for comparison."""
    host_compromised: Dict[str, int] = field(default_factory=dict)
    red_privilege: Dict[str, int] = field(default_factory=dict)
    red_sessions: Dict[str, int] = field(default_factory=dict)
    red_discovered_hosts: Dict[str, bool] = field(default_factory=dict)
    red_scanned_hosts: Dict[str, bool] = field(default_factory=dict)
    host_activity_detected: Dict[str, bool] = field(default_factory=dict)
    ot_service_stopped: Dict[str, bool] = field(default_factory=dict)
    blue_obs: Optional[np.ndarray] = None
    red_obs: Optional[np.ndarray] = None
    reward_blue: float = 0.0
    reward_red: float = 0.0
    last_red_action_success: bool = False
    time: int = 0


def extract_cyborg_state(cyborg_env, include_obs: bool = True) -> StateSnapshot:
    """Extract comprehensive state from CybORG environment.

    Args:
        cyborg_env: CybORG environment instance
        include_obs: Whether to include observations

    Returns:
        StateSnapshot with all comparable state
    """
    snapshot = StateSnapshot()

    for hostname in HOST_IDS.keys():
        snapshot.host_compromised[hostname] = COMPROMISE_NONE
        snapshot.red_privilege[hostname] = COMPROMISE_NONE
        snapshot.red_sessions[hostname] = 0
        snapshot.red_discovered_hosts[hostname] = False
        snapshot.red_scanned_hosts[hostname] = False
        snapshot.host_activity_detected[hostname] = False
        snapshot.ot_service_stopped[hostname] = False

    # Primary source: internal state sessions (most reliable)
    # CybORG's get_agent_state('Red') only returns partial info
    try:
        state = cyborg_env.environment_controller.state
        red_sessions_dict = state.sessions.get('Red', {})

        # Track which hosts have Red sessions
        hosts_with_sessions = set()
        for sess_id, sess in red_sessions_dict.items():
            hostname = sess.host
            if hostname in HOST_IDS:
                hosts_with_sessions.add(hostname)
                # Count sessions (avoid double-counting)
                if snapshot.red_sessions[hostname] == 0:
                    snapshot.red_sessions[hostname] = 1

                # Check privilege level
                username = sess.username
                if username in ['root', 'SYSTEM']:
                    snapshot.red_privilege[hostname] = COMPROMISE_PRIVILEGED
                    snapshot.host_compromised[hostname] = COMPROMISE_PRIVILEGED
                elif snapshot.red_privilege[hostname] < COMPROMISE_USER:
                    snapshot.red_privilege[hostname] = COMPROMISE_USER
                    if snapshot.host_compromised[hostname] < COMPROMISE_USER:
                        snapshot.host_compromised[hostname] = COMPROMISE_USER

        # CybORG doesn't explicitly track discovered/scanned hosts.
        # We mark hosts with sessions as "discovered" for comparison purposes,
        # but this won't match JAX's more aggressive discovery tracking.
        for hostname in hosts_with_sessions:
            snapshot.red_discovered_hosts[hostname] = True
            snapshot.red_scanned_hosts[hostname] = True

    except Exception:
        pass

    # Fallback: also check get_agent_state for any additional info
    red_state = cyborg_env.get_agent_state('Red')
    for hostname in HOST_IDS.keys():
        if hostname in red_state and isinstance(red_state[hostname], dict):
            host_info = red_state[hostname]

            if 'Interface' in host_info:
                snapshot.red_discovered_hosts[hostname] = True
            if 'Processes' in host_info or 'Services' in host_info:
                snapshot.red_scanned_hosts[hostname] = True

    rewards = cyborg_env.get_rewards()
    snapshot.reward_blue = rewards.get('Blue', 0.0)
    snapshot.reward_red = rewards.get('Red', 0.0)

    if include_obs:
        try:
            blue_obs_dict = cyborg_env.get_observation('Blue')
            snapshot.blue_obs = _convert_cyborg_blue_obs(blue_obs_dict)
        except Exception:
            pass

        try:
            red_obs_dict = cyborg_env.get_observation('Red')
            snapshot.red_obs = _convert_cyborg_red_obs(red_obs_dict)
        except Exception:
            pass

    return snapshot


def _convert_cyborg_blue_obs(obs_dict: dict) -> Optional[np.ndarray]:
    """Convert CybORG blue observation dict to array format."""
    if obs_dict is None:
        return None

    obs = np.zeros(len(HOST_IDS) * 4, dtype=np.float32)

    for hostname, host_idx in HOST_IDS.items():
        if hostname in obs_dict:
            host_data = obs_dict[hostname]

            activity = [0, 0]
            compromised = [0, 0]

            if isinstance(host_data, dict):
                if 'Processes' in host_data:
                    activity = [1, 0]
                if 'Sessions' in host_data:
                    activity = [1, 1]
                if host_data.get('System info', {}).get('Compromised') == 'User':
                    compromised = [0, 1]
                elif host_data.get('System info', {}).get('Compromised') == 'Privileged':
                    compromised = [1, 1]
                elif host_data.get('System info', {}).get('Compromised') == 'Unknown':
                    compromised = [1, 0]

            base_idx = host_idx * 4
            obs[base_idx:base_idx+2] = activity
            obs[base_idx+2:base_idx+4] = compromised

    return obs


def _convert_cyborg_red_obs(obs_dict: dict) -> Optional[np.ndarray]:
    """Convert CybORG red observation dict to array format."""
    if obs_dict is None:
        return None

    obs = np.zeros(1 + len(HOST_IDS) * 3, dtype=np.float32)

    success = obs_dict.get('success', True)
    obs[0] = 1.0 if success else 0.0

    for hostname, host_idx in HOST_IDS.items():
        base_idx = 1 + host_idx * 3
        if hostname in obs_dict:
            host_data = obs_dict[hostname]
            if isinstance(host_data, dict):
                if 'Processes' in host_data or 'Services' in host_data:
                    obs[base_idx] = 1.0
                sessions = host_data.get('Sessions', [])
                for session in sessions:
                    if isinstance(session, dict) and session.get('Agent') == 'Red':
                        username = session.get('Username', '')
                        if username in ['root', 'SYSTEM']:
                            obs[base_idx + 1] = 1.0
                            obs[base_idx + 2] = 1.0
                        else:
                            obs[base_idx + 1] = 1.0

    return obs


def extract_jax_state(state: CageState, const: CageConst, include_obs: bool = True) -> StateSnapshot:
    """Extract comprehensive state from CAGE-JAX environment.

    Args:
        state: CageState instance
        const: CageConst instance
        include_obs: Whether to include observations

    Returns:
        StateSnapshot with all comparable state
    """
    snapshot = StateSnapshot()

    for hostname, idx in HOST_IDS.items():
        snapshot.host_compromised[hostname] = int(state.host_compromised[idx])
        snapshot.red_privilege[hostname] = int(state.red_privilege[idx])
        snapshot.red_sessions[hostname] = int(state.red_sessions[idx])
        snapshot.red_discovered_hosts[hostname] = bool(state.red_discovered_hosts_jax[idx])
        snapshot.red_scanned_hosts[hostname] = bool(state.red_scanned_hosts_jax[idx])
        snapshot.host_activity_detected[hostname] = bool(state.host_activity_detected[idx])
        snapshot.ot_service_stopped[hostname] = bool(state.ot_service_stopped[idx])

    snapshot.last_red_action_success = bool(state.last_red_action_success)
    snapshot.time = int(state.time)

    if include_obs:
        snapshot.blue_obs = np.array(get_blue_obs(state, const))
        snapshot.red_obs = np.array(get_red_obs(state, const))

    from jaxmarl.environments.cage.rewards import compute_rewards_simple
    rewards = compute_rewards_simple(state, const)
    snapshot.reward_blue = float(rewards['blue'])
    snapshot.reward_red = float(rewards['red'])

    return snapshot


def compare_states(
    cyborg_state: StateSnapshot,
    jax_state: StateSnapshot,
    check_rewards: bool = True,
    check_obs: bool = False,
    reward_tolerance: float = 0.01,
) -> List[StateDiff]:
    """Compare CybORG and JAX states, returning list of differences.

    Args:
        cyborg_state: State snapshot from CybORG
        jax_state: State snapshot from JAX
        check_rewards: Whether to compare rewards
        check_obs: Whether to compare observations
        reward_tolerance: Tolerance for reward comparison

    Returns:
        List of StateDiff objects describing differences
    """
    diffs = []

    for hostname in HOST_IDS.keys():
        cyborg_comp = cyborg_state.host_compromised.get(hostname, 0)
        jax_comp = jax_state.host_compromised.get(hostname, 0)
        if cyborg_comp != jax_comp:
            diffs.append(StateDiff(
                field='host_compromised',
                host=hostname,
                cyborg_value=cyborg_comp,
                jax_value=jax_comp,
                severity='error',
            ))

        cyborg_priv = cyborg_state.red_privilege.get(hostname, 0)
        jax_priv = jax_state.red_privilege.get(hostname, 0)
        if cyborg_priv != jax_priv:
            diffs.append(StateDiff(
                field='red_privilege',
                host=hostname,
                cyborg_value=cyborg_priv,
                jax_value=jax_priv,
                severity='error',
            ))

        # Session presence is critical for game mechanics
        cyborg_sess = cyborg_state.red_sessions.get(hostname, 0)
        jax_sess = jax_state.red_sessions.get(hostname, 0)
        cyborg_has = cyborg_sess > 0
        jax_has = jax_sess > 0
        if cyborg_has != jax_has:
            diffs.append(StateDiff(
                field='red_sessions',
                host=hostname,
                cyborg_value=cyborg_sess,
                jax_value=jax_sess,
                severity='error',  # Sessions affect what actions are valid
            ))

        # Discovery/scan tracking: CybORG doesn't explicitly track these.
        # JAX tracks them for action masking, but CybORG computes validity differently.
        # We compare them as warnings to document differences, but they don't affect
        # core game mechanics (rewards, compromise state).
        cyborg_discovered = cyborg_state.red_discovered_hosts.get(hostname, False)
        jax_discovered = jax_state.red_discovered_hosts.get(hostname, False)
        if cyborg_discovered != jax_discovered:
            diffs.append(StateDiff(
                field='red_discovered_hosts',
                host=hostname,
                cyborg_value=cyborg_discovered,
                jax_value=jax_discovered,
                severity='warning',  # CybORG doesn't track this explicitly
            ))

        cyborg_scanned = cyborg_state.red_scanned_hosts.get(hostname, False)
        jax_scanned = jax_state.red_scanned_hosts.get(hostname, False)
        if cyborg_scanned != jax_scanned:
            diffs.append(StateDiff(
                field='red_scanned_hosts',
                host=hostname,
                cyborg_value=cyborg_scanned,
                jax_value=jax_scanned,
                severity='warning',  # CybORG doesn't track this explicitly
            ))

    if check_rewards:
        if abs(cyborg_state.reward_blue - jax_state.reward_blue) > reward_tolerance:
            diffs.append(StateDiff(
                field='reward_blue',
                host=None,
                cyborg_value=cyborg_state.reward_blue,
                jax_value=jax_state.reward_blue,
                severity='error',
            ))

        if abs(cyborg_state.reward_red - jax_state.reward_red) > reward_tolerance:
            diffs.append(StateDiff(
                field='reward_red',
                host=None,
                cyborg_value=cyborg_state.reward_red,
                jax_value=jax_state.reward_red,
                severity='error',
            ))

    if check_obs:
        if cyborg_state.blue_obs is not None and jax_state.blue_obs is not None:
            if not np.allclose(cyborg_state.blue_obs, jax_state.blue_obs, atol=0.01):
                diff_indices = np.where(np.abs(cyborg_state.blue_obs - jax_state.blue_obs) > 0.01)[0]
                diffs.append(StateDiff(
                    field='blue_obs',
                    host=None,
                    cyborg_value=f"differs at indices {diff_indices.tolist()}",
                    jax_value=f"values: cyborg={cyborg_state.blue_obs[diff_indices]}, jax={jax_state.blue_obs[diff_indices]}",
                    severity='warning',
                ))

        if cyborg_state.red_obs is not None and jax_state.red_obs is not None:
            if not np.allclose(cyborg_state.red_obs, jax_state.red_obs, atol=0.01):
                diff_indices = np.where(np.abs(cyborg_state.red_obs - jax_state.red_obs) > 0.01)[0]
                diffs.append(StateDiff(
                    field='red_obs',
                    host=None,
                    cyborg_value=f"differs at indices {diff_indices.tolist()}",
                    jax_value=f"values: cyborg={cyborg_state.red_obs[diff_indices]}, jax={jax_state.red_obs[diff_indices]}",
                    severity='warning',
                ))

    return diffs


def states_match(
    cyborg_state: StateSnapshot,
    jax_state: StateSnapshot,
    check_rewards: bool = True,
    check_obs: bool = False,
) -> bool:
    """Check if CybORG and JAX states match.

    Args:
        cyborg_state: State snapshot from CybORG
        jax_state: State snapshot from JAX
        check_rewards: Whether to compare rewards
        check_obs: Whether to compare observations

    Returns:
        True if states match (no error-level differences)
    """
    diffs = compare_states(cyborg_state, jax_state, check_rewards, check_obs)
    return not any(d.severity == 'error' for d in diffs)


def format_state_comparison(
    cyborg_state: StateSnapshot,
    jax_state: StateSnapshot,
    step: Optional[int] = None,
) -> str:
    """Format a comparison of two states for debugging output.

    Args:
        cyborg_state: State snapshot from CybORG
        jax_state: State snapshot from JAX
        step: Optional step number

    Returns:
        Formatted comparison string
    """
    lines = []
    step_str = f" (step {step})" if step is not None else ""
    lines.append(f"State Comparison{step_str}")
    lines.append("=" * 60)

    lines.append(f"\nRewards: CybORG blue={cyborg_state.reward_blue:.2f}, red={cyborg_state.reward_red:.2f}")
    lines.append(f"         JAX    blue={jax_state.reward_blue:.2f}, red={jax_state.reward_red:.2f}")

    lines.append("\nHost Compromise Status:")
    lines.append(f"{'Host':<15} {'CybORG':>12} {'JAX':>12} {'Match':>8}")
    lines.append("-" * 50)

    for hostname in sorted(HOST_IDS.keys()):
        cyborg_comp = cyborg_state.host_compromised.get(hostname, 0)
        jax_comp = jax_state.host_compromised.get(hostname, 0)
        match = "✓" if cyborg_comp == jax_comp else "✗"
        lines.append(f"{hostname:<15} {cyborg_comp:>12} {jax_comp:>12} {match:>8}")

    diffs = compare_states(cyborg_state, jax_state)
    if diffs:
        lines.append(f"\nDifferences ({len(diffs)}):")
        for diff in diffs:
            lines.append(f"  {diff}")

    return "\n".join(lines)


def compute_reward_deltas(
    prev_state: StateSnapshot,
    curr_state: StateSnapshot,
) -> Tuple[float, float]:
    """Compute reward deltas between two states.

    Args:
        prev_state: Previous state snapshot
        curr_state: Current state snapshot

    Returns:
        Tuple of (blue_delta, red_delta)
    """
    blue_delta = curr_state.reward_blue - prev_state.reward_blue
    red_delta = curr_state.reward_red - prev_state.reward_red
    return blue_delta, red_delta


def compare_reward_deltas(
    cyborg_prev: StateSnapshot,
    cyborg_curr: StateSnapshot,
    jax_prev: StateSnapshot,
    jax_curr: StateSnapshot,
    tolerance: float = 0.02,
) -> List[StateDiff]:
    """Compare reward deltas between CybORG and JAX.

    Args:
        cyborg_prev: Previous CybORG state
        cyborg_curr: Current CybORG state
        jax_prev: Previous JAX state
        jax_curr: Current JAX state
        tolerance: Tolerance for comparison

    Returns:
        List of StateDiff objects for reward delta mismatches
    """
    diffs = []

    cyborg_blue_delta, cyborg_red_delta = compute_reward_deltas(cyborg_prev, cyborg_curr)
    jax_blue_delta, jax_red_delta = compute_reward_deltas(jax_prev, jax_curr)

    if abs(cyborg_blue_delta - jax_blue_delta) > tolerance:
        diffs.append(StateDiff(
            field='reward_blue_delta',
            host=None,
            cyborg_value=cyborg_blue_delta,
            jax_value=jax_blue_delta,
            severity='error',
        ))

    if abs(cyborg_red_delta - jax_red_delta) > tolerance:
        diffs.append(StateDiff(
            field='reward_red_delta',
            host=None,
            cyborg_value=cyborg_red_delta,
            jax_value=jax_red_delta,
            severity='error',
        ))

    return diffs


def get_compromised_hosts(state: StateSnapshot) -> List[str]:
    """Get list of compromised hosts from state.

    Args:
        state: State snapshot

    Returns:
        List of hostname strings that are compromised (user or privileged)
    """
    compromised = []
    for hostname, level in state.host_compromised.items():
        if level >= COMPROMISE_USER:
            compromised.append(hostname)
    return compromised


def get_privileged_hosts(state: StateSnapshot) -> List[str]:
    """Get list of hosts where Red has privileged access.

    Args:
        state: State snapshot

    Returns:
        List of hostname strings with privileged Red access
    """
    privileged = []
    for hostname, level in state.red_privilege.items():
        if level >= COMPROMISE_PRIVILEGED:
            privileged.append(hostname)
    return privileged
