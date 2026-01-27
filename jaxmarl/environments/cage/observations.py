"""CAGE-JAX observation encoding with configurable support."""

import jax
import jax.numpy as jnp
import chex
from functools import partial

from jaxmarl.environments.cage.state import (
    CageState, CageConst,
    COMPROMISE_USER, COMPROMISE_PRIVILEGED,
)

# Default observation dimensions for backward compatibility (Scenario 2)
NUM_HOSTS = 13
BLUE_OBS_PER_HOST = 4
BLUE_OBS_DIM = NUM_HOSTS * BLUE_OBS_PER_HOST  # 52

RED_OBS_PER_HOST = 3
RED_OBS_DIM = 1 + NUM_HOSTS * RED_OBS_PER_HOST  # 40


def compute_blue_obs_dim(const: CageConst) -> int:
    """Compute blue observation dimension for given configuration."""
    return const.num_hosts * BLUE_OBS_PER_HOST


def compute_red_obs_dim(const: CageConst) -> int:
    """Compute red observation dimension for given configuration."""
    return 1 + const.num_hosts * RED_OBS_PER_HOST


def get_blue_obs(state: CageState, const: CageConst) -> chex.Array:
    """Get Blue agent observation matching CybORG's encoding.

    CybORG encoding per host (4 values):
        Activity:    None=[0,0], Scan=[1,0], Exploit=[1,1]
        Compromised: No=[0,0],   User=[0,1], Priv=[1,1], Unknown=[1,0]

    CybORG's BlueTableWrapper automatically detects anomalies by comparing
    current host state (processes/files) to baseline. This means:
    - Red's presence (sessions) creates detectable anomalies
    - Anomalies persist until Blue does Restore (which resets to baseline)
    - Blue does NOT need to explicitly Monitor to see anomalies

    Red's initial foothold (User0) is hidden because it's part of the baseline.
    After Remove action, compromised shows "Unknown" [1,0] until Restore.

    Returns:
        Array of shape (num_hosts * 4,) with per-host features:
        [activity_0, activity_1, compromised_0, compromised_1]
    """
    # Red's initial foothold host (User0) is hidden from Blue
    # This is because in CybORG, the initial state IS the baseline
    # Use zeros_like for JIT compatibility (can't use const.num_hosts directly)
    initial_foothold_mask = jnp.zeros_like(state.red_sessions, dtype=jnp.bool_)
    initial_foothold_mask = initial_foothold_mask.at[const.red_start_hosts].set(True)

    # Activity encoding: None=[0,0], Scan=[1,0], Exploit=[1,1]
    # Blue sees activity when Red has sessions (persistent anomaly)
    # or when recent activity was detected via Monitor
    has_red_presence = state.red_sessions > 0
    recent_scan = state.red_activity_this_step & ~has_red_presence

    # Hide initial foothold from observation (matches CybORG baseline behavior)
    visible_presence = has_red_presence & ~initial_foothold_mask

    # activity_0 = 1 if any activity (scan or exploit)
    # activity_1 = 1 if exploit (Red has session)
    activity_0 = (visible_presence | recent_scan).astype(jnp.float32)
    activity_1 = visible_presence.astype(jnp.float32)

    # Compromised encoding: No=[0,0], User=[0,1], Priv=[1,1], Unknown=[1,0]
    # Blue sees compromise level for hosts where Red has visible presence
    compromised_0 = (visible_presence & (state.host_compromised >= COMPROMISE_PRIVILEGED)).astype(jnp.float32)
    compromised_1 = (visible_presence & (state.host_compromised >= COMPROMISE_USER)).astype(jnp.float32)

    # After Remove, compromised shows Unknown [1,0] until Restore
    compromised_0 = jnp.where(state.host_observation_unknown, 1.0, compromised_0)
    compromised_1 = jnp.where(state.host_observation_unknown, 0.0, compromised_1)

    obs = jnp.stack([activity_0, activity_1, compromised_0, compromised_1], axis=1)
    return obs.flatten()


def get_red_obs(state: CageState, const: CageConst) -> chex.Array:
    """Get Red agent observation.

    Returns:
        Array of shape (1 + num_hosts * 3,) with:
        - success_flag: whether last action succeeded
        - Per-host features:
          - scanned: host has been port scanned
          - user_access: Red has user-level access
          - privileged_access: Red has privileged access
    """
    scanned = state.red_scanned_hosts_jax.astype(jnp.float32)
    user_access = (state.red_privilege >= COMPROMISE_USER).astype(jnp.float32)
    privileged_access = (state.red_privilege >= COMPROMISE_PRIVILEGED).astype(jnp.float32)

    host_obs = jnp.stack([scanned, user_access, privileged_access], axis=1).flatten()
    success_flag = state.last_red_action_success.astype(jnp.float32).reshape(1)

    return jnp.concatenate([success_flag, host_obs])


def get_obs(state: CageState, const: CageConst) -> dict[str, chex.Array]:
    """Get observations for both agents.

    Returns:
        Dict with 'blue' and 'red' observation arrays.
    """
    return {
        'blue': get_blue_obs(state, const),
        'red': get_red_obs(state, const),
    }
