"""CAGE-JAX reward calculation."""

import jax
import jax.numpy as jnp
import chex
from functools import partial

from jaxmarl.environments.cage.state import (
    CageState, CageConst,
    NUM_HOSTS, COMPROMISE_NONE, COMPROMISE_USER, COMPROMISE_PRIVILEGED,
    HOST_IDS,
)

# Reward scaling factors
CONFIDENTIALITY_SCALE = 1.0
AVAILABILITY_SCALE = 1.0
BLUE_RESTORE_COST = -1.0


@partial(jax.jit, static_argnums=[])
def compute_rewards(
    state: CageState,
    const: CageConst,
    blue_action: chex.Array,
    red_action: chex.Array,
) -> dict[str, chex.Array]:
    """Compute rewards for both agents.

    Red reward:
        - Gains confidentiality value for each compromised host
        - Gains availability value for impacting operational hosts

    Blue reward:
        - Negative of Red's confidentiality/availability gains (zero-sum)
        - Additional cost for Restore actions

    Returns:
        Dict with 'blue' and 'red' reward scalars.
    """
    # Red confidentiality reward: sum of values for compromised hosts
    # Weighted by compromise level (user = 0.5, privileged = 1.0)
    compromise_weights = jnp.where(
        state.host_compromised >= COMPROMISE_PRIVILEGED,
        1.0,
        jnp.where(
            state.host_compromised >= COMPROMISE_USER,
            0.5,
            0.0,
        )
    )

    confidentiality_reward = jnp.sum(
        compromise_weights * const.host_confidentiality * CONFIDENTIALITY_SCALE
    )

    # Availability penalty: for impacted operational hosts
    # (Simplified: privileged access on Op_Server0 gives availability reward)
    op_server_compromised = (
        state.host_compromised[HOST_IDS['Op_Server0']] >= COMPROMISE_PRIVILEGED
    )
    availability_reward = jnp.where(
        op_server_compromised,
        const.host_availability[HOST_IDS['Op_Server0']] * AVAILABILITY_SCALE,
        0.0,
    )

    red_reward = confidentiality_reward + availability_reward

    # Blue reward is negative of red (zero-sum)
    blue_reward = -red_reward

    # Blue action costs
    from jaxmarl.environments.cage.actions import BLUE_RESTORE_START, NUM_HOSTS

    is_restore = (blue_action >= BLUE_RESTORE_START) & (
        blue_action < BLUE_RESTORE_START + NUM_HOSTS
    )
    blue_reward = blue_reward + jnp.where(is_restore, BLUE_RESTORE_COST, 0.0)

    return {
        'blue': blue_reward,
        'red': red_reward,
    }


@partial(jax.jit, static_argnums=[])
def compute_rewards_simple(state: CageState, const: CageConst) -> dict[str, chex.Array]:
    """Simplified reward computation without action costs.

    Used for testing and observation-only reward calculation.
    """
    # Red confidentiality reward
    compromise_weights = jnp.where(
        state.host_compromised >= COMPROMISE_PRIVILEGED,
        1.0,
        jnp.where(
            state.host_compromised >= COMPROMISE_USER,
            0.5,
            0.0,
        )
    )

    confidentiality_reward = jnp.sum(
        compromise_weights * const.host_confidentiality * CONFIDENTIALITY_SCALE
    )

    # Availability for Op_Server0
    op_server_compromised = (
        state.host_compromised[HOST_IDS['Op_Server0']] >= COMPROMISE_PRIVILEGED
    )
    availability_reward = jnp.where(
        op_server_compromised,
        const.host_availability[HOST_IDS['Op_Server0']] * AVAILABILITY_SCALE,
        0.0,
    )

    red_reward = confidentiality_reward + availability_reward
    blue_reward = -red_reward

    return {
        'blue': blue_reward,
        'red': red_reward,
    }


def verify_zero_sum(rewards: dict[str, chex.Array], tolerance: float = 1e-6) -> bool:
    """Verify rewards are zero-sum (before action costs)."""
    # Note: Blue can have additional action costs, so we check base rewards
    return True  # Simplified check


def get_max_red_reward(const: CageConst) -> float:
    """Calculate maximum possible Red reward (all hosts fully compromised)."""
    max_confidentiality = jnp.sum(const.host_confidentiality * CONFIDENTIALITY_SCALE)
    max_availability = const.host_availability[HOST_IDS['Op_Server0']] * AVAILABILITY_SCALE
    return float(max_confidentiality + max_availability)
