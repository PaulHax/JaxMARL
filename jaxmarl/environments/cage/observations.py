"""CAGE-JAX observation encoding with configurable support."""

import jax
import jax.numpy as jnp
import chex
from functools import partial

from jaxmarl.environments.cage.state import (
    CageState, CageConst,
    COMPROMISE_USER, COMPROMISE_PRIVILEGED,
    ACTIVITY_NONE, ACTIVITY_SCAN, ACTIVITY_EXPLOIT,
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

    CybORG's BlueTableWrapper behavior:
    - Activity is TRANSIENT: based on anomalous processes that appear THIS step
      (DiscoverNetworkServices → Scan, ExploitRemoteService → Exploit)
    - After PrivilegeEscalate or other actions, processes are cleared so activity=None
    - Compromise is PERSISTENT: stays as long as Red has a session on the host

    Red's initial foothold (User0) is hidden because it's part of the baseline.
    After Remove action, compromised shows "Unknown" [1,0] until Restore.

    Returns:
        Array of shape (num_hosts * 4,) with per-host features:
        [activity_0, activity_1, compromised_0, compromised_1]
    """
    # Red's initial foothold host (User0) is hidden from Blue
    initial_foothold_mask = jnp.zeros_like(state.red_sessions, dtype=jnp.bool_)
    initial_foothold_mask = initial_foothold_mask.at[const.red_start_hosts].set(True)

    has_session = state.red_sessions > 0

    # Activity encoding: None=[0,0], Scan=[1,0], Exploit=[1,1]
    # CybORG activity is TRANSIENT - only shows when specific actions occur THIS step
    # red_activity_this_step: 0=None, 1=Scan, 2=Exploit
    activity_type = state.red_activity_this_step
    has_scan = (activity_type == ACTIVITY_SCAN) & ~initial_foothold_mask
    has_exploit = (activity_type == ACTIVITY_EXPLOIT) & ~initial_foothold_mask

    # Scan=[1,0], Exploit=[1,1], None=[0,0]
    activity_0 = (has_scan | has_exploit).astype(jnp.float32)
    activity_1 = has_exploit.astype(jnp.float32)

    # Compromised encoding: No=[0,0], User=[0,1], Priv=[1,1], Unknown=[1,0]
    # CybORG's BlueTableWrapper detects:
    # - User: when shell connection detected (port 4444) → has_session
    # - Privileged: when malware detected (density >= 0.9) → host_has_malware
    # Note: Red may have higher actual privilege than Blue can observe
    visible_session = has_session & ~initial_foothold_mask
    has_malware = state.host_has_malware & ~initial_foothold_mask
    compromised_0 = has_malware.astype(jnp.float32)
    compromised_1 = visible_session.astype(jnp.float32)

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
