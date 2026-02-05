"""ResilienceMetric rewards for CAGE-JAX.

Implements CybORG's ResilienceMetric which awards Blue bonuses for taking
correct defensive actions in response to Red's successful attacks.

Bonus conditions:
- After successful Exploit: Blue should Remove the exploited host
- After successful PrivEsc/Impact: Blue should Restore the affected host

Host types have different CIA (Confidentiality, Integrity, Availability) scores:
- Auth: C=10, I=10, A=10
- Database: C=10, I=0, A=10
- Front: C=0, I=10, A=10
- Normal (all others): no resilience bonus

The weighted score is: C*0.6 + I*0.2 + A*0.2, normalized by max (10.0).
"""

import jax.numpy as jnp
import chex

from jaxmarl.environments.cage.state import (
    CageState, CageConst,
    HOST_TYPE_NORMAL, HOST_TYPE_AUTH, HOST_TYPE_DATABASE, HOST_TYPE_FRONT,
)

# CIA scores for each host type (Confidentiality, Integrity, Availability)
# From CybORG's scorer_mappings.py
AUTH_CIA = (10.0, 10.0, 10.0)
DATABASE_CIA = (10.0, 0.0, 10.0)
FRONT_CIA = (0.0, 10.0, 10.0)
NORMAL_CIA = (0.0, 0.0, 0.0)

# Weights matching CybORG's ResilienceMetric
CONF_WEIGHT = 0.6
INTEG_WEIGHT = 0.2
AVAIL_WEIGHT = 0.2

# Maximum possible weighted score (for normalization)
MAX_WEIGHTED_SCORE = 10.0

# Red action types that can trigger resilience bonuses
RED_ACTION_EXPLOIT = 3
RED_ACTION_PRIVESC = 4
RED_ACTION_IMPACT = 5

# Blue action types
BLUE_ACTION_REMOVE = 3
BLUE_ACTION_RESTORE = 4


def get_cia_scores(host_type: chex.Array) -> tuple[chex.Array, chex.Array, chex.Array]:
    """Get CIA scores based on host type.

    Args:
        host_type: Host type (0=Normal, 1=Auth, 2=Database, 3=Front)

    Returns:
        Tuple of (confidentiality, integrity, availability) scores
    """
    confidentiality = jnp.where(
        host_type == HOST_TYPE_AUTH, AUTH_CIA[0],
        jnp.where(
            host_type == HOST_TYPE_DATABASE, DATABASE_CIA[0],
            jnp.where(
                host_type == HOST_TYPE_FRONT, FRONT_CIA[0],
                NORMAL_CIA[0]
            )
        )
    )

    integrity = jnp.where(
        host_type == HOST_TYPE_AUTH, AUTH_CIA[1],
        jnp.where(
            host_type == HOST_TYPE_DATABASE, DATABASE_CIA[1],
            jnp.where(
                host_type == HOST_TYPE_FRONT, FRONT_CIA[1],
                NORMAL_CIA[1]
            )
        )
    )

    availability = jnp.where(
        host_type == HOST_TYPE_AUTH, AUTH_CIA[2],
        jnp.where(
            host_type == HOST_TYPE_DATABASE, DATABASE_CIA[2],
            jnp.where(
                host_type == HOST_TYPE_FRONT, FRONT_CIA[2],
                NORMAL_CIA[2]
            )
        )
    )

    return confidentiality, integrity, availability


def compute_weighted_score(c: chex.Array, i: chex.Array, a: chex.Array) -> chex.Array:
    """Compute weighted CIA score.

    Args:
        c: Confidentiality score
        i: Integrity score
        a: Availability score

    Returns:
        Weighted score: C*0.6 + I*0.2 + A*0.2
    """
    return c * CONF_WEIGHT + i * INTEG_WEIGHT + a * AVAIL_WEIGHT


def compute_resilience_bonus(
    state: CageState,
    const: CageConst,
    blue_action_type: chex.Array,
    blue_action_target: chex.Array,
) -> chex.Array:
    """Compute resilience bonus for Blue's defensive action.

    Awards bonus when Blue takes the correct defensive action in response
    to Red's successful action:
    - After successful Exploit: Blue should Remove the exploited host
    - After successful PrivEsc/Impact: Blue should Restore the affected host

    Args:
        state: Current game state (includes last_red_action_* fields)
        const: Environment constants (includes host_type)
        blue_action_type: Blue's action type (from decode_blue_action)
        blue_action_target: Blue's target host

    Returns:
        Resilience bonus (0.0 to 1.0, or 0.0 if no bonus applies)
    """
    # Get Red's last action info
    red_action_type = state.last_red_action_type
    red_target = state.last_red_action_target
    red_success = state.last_red_action_success

    # Check if Red's action was successful and on a valid target
    red_had_successful_action = red_success & (red_target >= 0)

    # Get host type and CIA scores for Red's target
    # Use safe indexing (clamp to valid range)
    safe_target = jnp.clip(red_target, 0, const.num_hosts - 1)
    host_type = const.host_type[safe_target]
    c, i, a = get_cia_scores(host_type)
    weighted_score = compute_weighted_score(c, i, a)
    normalized_score = weighted_score / MAX_WEIGHTED_SCORE

    # Check if Blue's action is the correct response
    # Correct response to Exploit: Remove the same host
    correct_remove = (
        (red_action_type == RED_ACTION_EXPLOIT) &
        (blue_action_type == BLUE_ACTION_REMOVE) &
        (blue_action_target == red_target)
    )

    # Correct response to PrivEsc or Impact: Restore the same host
    correct_restore = (
        ((red_action_type == RED_ACTION_PRIVESC) | (red_action_type == RED_ACTION_IMPACT)) &
        (blue_action_type == BLUE_ACTION_RESTORE) &
        (blue_action_target == red_target)
    )

    # Award bonus if correct response to successful Red action
    is_correct_response = correct_remove | correct_restore
    bonus = jnp.where(
        red_had_successful_action & is_correct_response,
        normalized_score,
        0.0
    )

    return bonus


def has_resilience_hosts(const: CageConst) -> bool:
    """Check if the scenario has any resilience-eligible hosts (Auth/Database/Front).

    Used to determine whether to enable resilience rewards.
    """
    has_special_hosts = jnp.any(const.host_type > HOST_TYPE_NORMAL)
    return bool(has_special_hosts)
