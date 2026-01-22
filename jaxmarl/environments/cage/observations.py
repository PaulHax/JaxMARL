"""CAGE-JAX observation encoding."""

import jax
import jax.numpy as jnp
import chex
from functools import partial

from jaxmarl.environments.cage.state import (
    CageState, CageConst,
    NUM_HOSTS, COMPROMISE_USER, COMPROMISE_PRIVILEGED,
)

# Blue observation: 4 features per host = 52 dims
# [scan_detected, exploit_detected, user_compromised, privileged_compromised] × 13 hosts
BLUE_OBS_PER_HOST = 4
BLUE_OBS_DIM = NUM_HOSTS * BLUE_OBS_PER_HOST  # 52

# Red observation: 1 success flag + 3 features per host = 40 dims
# success_flag + [scanned, user_access, privileged_access] × 13 hosts
RED_OBS_PER_HOST = 3
RED_OBS_DIM = 1 + NUM_HOSTS * RED_OBS_PER_HOST  # 40


@partial(jax.jit, static_argnums=[])
def get_blue_obs(state: CageState) -> chex.Array:
    """Get Blue agent observation.

    Returns:
        Array of shape (52,) with per-host features:
        - scan_detected: activity indicator (simplified: any red session)
        - exploit_detected: activity indicator (simplified: any compromise)
        - user_compromised: host has user-level compromise
        - privileged_compromised: host has privileged compromise
    """
    obs = jnp.zeros((NUM_HOSTS, BLUE_OBS_PER_HOST), dtype=jnp.float32)

    # Detection based on red sessions (simplified detection model)
    # In full CybORG, this depends on Monitor action and probabilistic detection
    scan_detected = (state.red_scanned_hosts).astype(jnp.float32)
    exploit_detected = (state.red_sessions > 0).astype(jnp.float32)

    # Compromise levels
    user_compromised = (state.host_compromised >= COMPROMISE_USER).astype(jnp.float32)
    privileged_compromised = (state.host_compromised >= COMPROMISE_PRIVILEGED).astype(jnp.float32)

    obs = obs.at[:, 0].set(scan_detected)
    obs = obs.at[:, 1].set(exploit_detected)
    obs = obs.at[:, 2].set(user_compromised)
    obs = obs.at[:, 3].set(privileged_compromised)

    return obs.flatten()


@partial(jax.jit, static_argnums=[])
def get_red_obs(state: CageState) -> chex.Array:
    """Get Red agent observation.

    Returns:
        Array of shape (40,) with:
        - success_flag: whether last action succeeded
        - Per-host features (×13):
          - scanned: host has been port scanned
          - user_access: Red has user-level access
          - privileged_access: Red has privileged access
    """
    obs = jnp.zeros(RED_OBS_DIM, dtype=jnp.float32)

    # Success flag from last action
    obs = obs.at[0].set(state.last_red_action_success.astype(jnp.float32))

    # Per-host features
    for i in range(NUM_HOSTS):
        base_idx = 1 + i * RED_OBS_PER_HOST

        # Host scanned
        obs = obs.at[base_idx].set(state.red_scanned_hosts[i].astype(jnp.float32))

        # User access
        obs = obs.at[base_idx + 1].set(
            (state.red_privilege[i] >= COMPROMISE_USER).astype(jnp.float32)
        )

        # Privileged access
        obs = obs.at[base_idx + 2].set(
            (state.red_privilege[i] >= COMPROMISE_PRIVILEGED).astype(jnp.float32)
        )

    return obs


def get_obs(state: CageState) -> dict[str, chex.Array]:
    """Get observations for both agents.

    Returns:
        Dict with 'blue' and 'red' observation arrays.
    """
    return {
        'blue': get_blue_obs(state),
        'red': get_red_obs(state),
    }
