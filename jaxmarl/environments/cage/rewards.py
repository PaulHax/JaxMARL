"""CAGE-JAX reward calculation with configurable support."""

import jax
import jax.numpy as jnp
import chex
from functools import partial

from jaxmarl.environments.cage.state import (
    CageState, CageConst,
    COMPROMISE_PRIVILEGED,
)

CONFIDENTIALITY_SCALE = 1.0
AVAILABILITY_SCALE = 1.0
BLUE_RESTORE_COST = -1.0  # CybORG Restore.cost = -1
BLUE_INVALID_ACTION_COST = -0.1  # CybORG InvalidAction.cost = -0.1


def compute_rewards(
    state: CageState,
    const: CageConst,
    blue_action: chex.Array,
    red_action: chex.Array,
) -> dict[str, chex.Array]:
    """Compute rewards for both agents.

    CybORG-equivalent reward calculation:
    - Red only gets confidentiality for PRIVILEGED (root/SYSTEM) sessions
    - Red gets availability only for operational hosts where Impact has stopped OT service
    - Blue reward is negative of Red's (zero-sum base)
    - Blue pays additional cost for Restore actions (CybORG Restore.cost = -1)
    - Blue pays -0.1 penalty for failed Remove actions only
      (CybORG does NOT penalize failed Decoys - they fail silently with cost=0)

    Returns:
        Dict with 'blue' and 'red' reward scalars.
    """
    privileged_hosts = (state.host_compromised >= COMPROMISE_PRIVILEGED).astype(jnp.float32)

    confidentiality_reward = jnp.sum(
        privileged_hosts * const.host_confidentiality * CONFIDENTIALITY_SCALE
    )

    # Availability: only counts where OT service was stopped by Impact action
    ot_stopped = state.ot_service_stopped.astype(jnp.float32)
    availability_reward = jnp.sum(
        ot_stopped * const.host_availability * AVAILABILITY_SCALE
    )

    red_reward = confidentiality_reward + availability_reward
    blue_reward = -red_reward

    # Blue action costs (CybORG Restore.cost = -1)
    from jaxmarl.environments.cage.actions import get_blue_action_offsets, compute_blue_action_space_size

    analyse_start, remove_start, decoy_start, restore_start = get_blue_action_offsets(const)
    action_space_size = compute_blue_action_space_size(const)

    # Restore actions are at the end: [restore_start, action_space_size)
    is_restore = (blue_action >= restore_start) & (blue_action < action_space_size)
    blue_reward = blue_reward + jnp.where(is_restore, BLUE_RESTORE_COST, 0.0)

    # Failed action penalty (CybORG InvalidAction.cost = -0.1)
    # Only applied for failed Remove actions - CybORG does NOT penalize failed Decoys
    # (Decoys fail silently with cost=0 when port/OS incompatible)
    is_remove = (blue_action >= remove_start) & (blue_action < decoy_start)
    remove_failed = is_remove & ~state.last_blue_action_success
    blue_reward = blue_reward + jnp.where(remove_failed, BLUE_INVALID_ACTION_COST, 0.0)

    return {
        'blue': blue_reward,
        'red': red_reward,
    }


def compute_rewards_simple(state: CageState, const: CageConst) -> dict[str, chex.Array]:
    """Simplified reward computation without action costs."""
    privileged_hosts = (state.host_compromised >= COMPROMISE_PRIVILEGED).astype(jnp.float32)

    confidentiality_reward = jnp.sum(
        privileged_hosts * const.host_confidentiality * CONFIDENTIALITY_SCALE
    )

    # Availability: only counts where OT service was stopped by Impact action
    ot_stopped = state.ot_service_stopped.astype(jnp.float32)
    availability_reward = jnp.sum(
        ot_stopped * const.host_availability * AVAILABILITY_SCALE
    )

    red_reward = confidentiality_reward + availability_reward
    blue_reward = -red_reward

    return {
        'blue': blue_reward,
        'red': red_reward,
    }


def get_max_red_reward(const: CageConst) -> float:
    """Calculate maximum possible Red reward."""
    max_confidentiality = jnp.sum(const.host_confidentiality * CONFIDENTIALITY_SCALE)
    max_availability = jnp.sum(
        const.operational_targets.astype(jnp.float32) * const.host_availability * AVAILABILITY_SCALE
    )
    return float(max_confidentiality + max_availability)
