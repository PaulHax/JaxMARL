"""CAGE-JAX reward calculation with configurable support."""

import jax
import jax.numpy as jnp
import chex
from functools import partial

from jaxmarl.environments.cage.state import (
    CageState, CageConst,
    COMPROMISE_PRIVILEGED,
    HOST_TYPE_NORMAL,
)
from jaxmarl.environments.cage.resilience_rewards import (
    compute_resilience_bonus,
)

CONFIDENTIALITY_SCALE = 1.0
AVAILABILITY_SCALE = 1.0
BLUE_RESTORE_COST = -1.0  # CybORG Restore.cost = -1
DEFAULT_RESILIENCE_GAMMA = 1.0  # Default weight for resilience bonus


def compute_rewards(
    state: CageState,
    const: CageConst,
    blue_action: chex.Array,
    red_action: chex.Array,
    resilience_gamma: float = 0.0,
) -> dict[str, chex.Array]:
    """Compute rewards for both agents.

    CybORG-equivalent reward calculation:
    - Red only gets confidentiality for PRIVILEGED (root/SYSTEM) sessions
    - Red gets availability only for operational hosts where Impact has stopped OT service
    - Blue reward is negative of Red's (zero-sum base)
    - Blue pays additional cost for Restore actions (CybORG Restore.cost = -1)
    - CybORG does NOT penalize failed Remove or Decoy actions (both have cost=0)

    When resilience_gamma > 0, adds resilience bonus for correct Blue responses:
    - After Red Exploit: Blue Remove on same host gives bonus
    - After Red PrivEsc/Impact: Blue Restore on same host gives bonus

    Args:
        state: Current game state
        const: Environment constants
        blue_action: Blue's action index
        red_action: Red's action index
        resilience_gamma: Weight for resilience bonus (0.0 = disabled)

    Returns:
        Dict with 'blue' and 'red' reward scalars.
    """
    # CybORG only counts "valid" privileged sessions for confidentiality:
    # - SYSTEM on Windows = valid
    # - root on Linux = valid (from PrivEsc)
    # - SYSTEM on Linux = NOT valid (abstract session from initial foothold)
    # host_has_valid_privesc tracks this exactly.
    valid_privileged = state.host_has_valid_privesc.astype(jnp.float32)

    confidentiality_reward = jnp.sum(
        valid_privileged * const.host_confidentiality * CONFIDENTIALITY_SCALE
    )

    # Availability: only counts where OT service was stopped by Impact action
    ot_stopped = state.ot_service_stopped.astype(jnp.float32)
    availability_reward = jnp.sum(
        ot_stopped * const.host_availability * AVAILABILITY_SCALE
    )

    red_reward = confidentiality_reward + availability_reward
    blue_reward = -red_reward

    # Blue action costs (CybORG Restore.cost = -1)
    from jaxmarl.environments.cage.actions import (
        get_blue_action_offsets, compute_blue_action_space_size, decode_blue_action
    )

    analyse_start, remove_start, decoy_start, restore_start = get_blue_action_offsets(const)
    action_space_size = compute_blue_action_space_size(const)

    # Restore actions are at the end: [restore_start, action_space_size)
    is_restore = (blue_action >= restore_start) & (blue_action < action_space_size)
    blue_reward = blue_reward + jnp.where(is_restore, BLUE_RESTORE_COST, 0.0)

    # Add resilience bonus if enabled
    if resilience_gamma > 0.0:
        blue_action_type, blue_action_target, _ = decode_blue_action(blue_action, const)
        resilience_bonus = compute_resilience_bonus(
            state, const, blue_action_type, blue_action_target
        )
        blue_reward = blue_reward + resilience_gamma * resilience_bonus

    return {
        'blue': blue_reward,
        'red': red_reward,
    }


def compute_rewards_simple(state: CageState, const: CageConst) -> dict[str, chex.Array]:
    """Simplified reward computation without action costs."""
    valid_privileged = state.host_has_valid_privesc.astype(jnp.float32)

    confidentiality_reward = jnp.sum(
        valid_privileged * const.host_confidentiality * CONFIDENTIALITY_SCALE
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
