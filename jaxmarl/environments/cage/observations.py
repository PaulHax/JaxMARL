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
    """Get Blue agent observation.

    Returns:
        Array of shape (num_hosts * 4,) with per-host features:
        - scan_detected: activity indicator
        - exploit_detected: activity indicator
        - user_compromised: host has user-level compromise
        - privileged_compromised: host has privileged compromise
    """
    scan_detected = (state.red_scanned_hosts).astype(jnp.float32)
    exploit_detected = (state.red_sessions > 0).astype(jnp.float32)
    user_compromised = (state.host_compromised >= COMPROMISE_USER).astype(jnp.float32)
    privileged_compromised = (state.host_compromised >= COMPROMISE_PRIVILEGED).astype(jnp.float32)

    obs = jnp.stack([scan_detected, exploit_detected, user_compromised, privileged_compromised], axis=1)
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
    scanned = state.red_scanned_hosts.astype(jnp.float32)
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
