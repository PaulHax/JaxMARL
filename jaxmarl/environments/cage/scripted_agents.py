"""Scripted agents for CAGE-JAX environment.

Implements fixed-behavior agents matching CybORG's B_lineAgent for
training Blue agents against deterministic Red strategies.
"""

import jax
import jax.numpy as jnp
from flax import struct
import chex
from typing import Tuple

from jaxmarl.environments.cage.state import CageConst
from jaxmarl.environments.cage.actions import get_red_action_offsets


# B_lineAgent target hosts for Scenario2 (matching CybORG)
# Host indices: User0=0, User1=1, User2=2, User3=3, User4=4
#              Enterprise0=5, Enterprise1=6, Enterprise2=7, Defender=8
#              Op_Host0=9, Op_Host1=10, Op_Host2=11, Op_Server0=12
BLINE_USER_SUBNET = 0
BLINE_ENTERPRISE_SUBNET = 1
BLINE_OPERATIONAL_SUBNET = 2
BLINE_USER_HOST = 1  # User1 - initial target in User subnet
BLINE_ENTERPRISE0 = 5
BLINE_ENTERPRISE2 = 7
BLINE_OP_SERVER0 = 12

# Jump-back table: on failure at state i, jump to state BLINE_JUMP_BACK[i]
# FSM has 16 states (0-15) with DiscoverSubnet calls inserted
BLINE_JUMP_BACK = jnp.array([
    0,   # State 0:  DiscoverSubnet(User) -> retry
    1,   # State 1:  ScanHost(User1) -> retry
    2,   # State 2:  Exploit(User1) -> retry
    2,   # State 3:  PrivEsc(User1) -> back to exploit
    4,   # State 4:  DiscoverSubnet(Enterprise) -> retry
    5,   # State 5:  ScanHost(Enterprise0) -> retry
    6,   # State 6:  Exploit(Enterprise0) -> retry
    6,   # State 7:  PrivEsc(Enterprise0) -> back to exploit
    8,   # State 8:  ScanHost(Enterprise2) -> retry
    9,   # State 9:  Exploit(Enterprise2) -> retry
    9,   # State 10: PrivEsc(Enterprise2) -> back to exploit
    11,  # State 11: DiscoverSubnet(Operational) -> retry
    12,  # State 12: ScanHost(Op_Server0) -> retry
    13,  # State 13: Exploit(Op_Server0) -> retry
    13,  # State 14: PrivEsc(Op_Server0) -> back to exploit
    14,  # State 15: Impact(Op_Server0) -> back to privesc
])

# Default exploit type for B_lineAgent (SSH brute force = 0)
BLINE_EXPLOIT_TYPE = 0

# Max FSM state
BLINE_MAX_STATE = 15


@struct.dataclass
class BLineState:
    """State for B_lineAgent finite state machine."""
    fsm_state: chex.Array  # Current FSM state (0-15)
    last_action_success: chex.Array  # Track last action result


def bline_reset() -> BLineState:
    """Reset B_lineAgent state to initial FSM state 0."""
    return BLineState(
        fsm_state=jnp.array(0, dtype=jnp.int32),
        last_action_success=jnp.array(True, dtype=jnp.bool_),
    )


def bline_get_action(
    agent_state: BLineState,
    red_obs: chex.Array,
    action_mask: chex.Array,
    const: CageConst,
    key: chex.PRNGKey,
) -> Tuple[chex.Array, BLineState]:
    """Get B_lineAgent action based on FSM state.

    The B_lineAgent follows a deterministic 16-state FSM:
    - State 0:  DiscoverSubnet(User)
    - State 1:  ScanHost(User1)
    - State 2:  Exploit(User1)
    - State 3:  PrivEsc(User1)
    - State 4:  DiscoverSubnet(Enterprise)
    - State 5:  ScanHost(Enterprise0)
    - State 6:  Exploit(Enterprise0)
    - State 7:  PrivEsc(Enterprise0)
    - State 8:  ScanHost(Enterprise2)
    - State 9:  Exploit(Enterprise2)
    - State 10: PrivEsc(Enterprise2)
    - State 11: DiscoverSubnet(Operational)
    - State 12: ScanHost(Op_Server0)
    - State 13: Exploit(Op_Server0)
    - State 14: PrivEsc(Op_Server0)
    - State 15: Impact(Op_Server0)

    On action failure, jumps back using BLINE_JUMP_BACK table.
    On success, advances to next state (capped at 15).

    Args:
        agent_state: Current B_lineAgent FSM state
        red_obs: Red agent observation (first element is last_action_success)
        action_mask: Valid action mask
        const: Environment constants
        key: Random key (unused, for API compatibility)

    Returns:
        Tuple of (action_index, new_agent_state)
    """
    # Extract success flag from observation (index 0)
    last_success = red_obs[0] > 0.5

    # Update FSM state based on last action result
    current_state = agent_state.fsm_state

    # On success: advance to next state (cap at BLINE_MAX_STATE)
    # On failure: jump back using table
    new_fsm_state = jax.lax.cond(
        last_success,
        lambda s: jnp.minimum(s + 1, BLINE_MAX_STATE),
        lambda s: BLINE_JUMP_BACK[s],
        current_state,
    )

    # Get action for the new FSM state
    action = _fsm_state_to_action(new_fsm_state, const)

    # Ensure action is valid (fallback to Sleep=0 if invalid)
    action = jax.lax.cond(
        action_mask[action],
        lambda: action,
        lambda: jnp.array(0, dtype=jnp.int32),
    )

    new_agent_state = BLineState(
        fsm_state=new_fsm_state,
        last_action_success=last_success,
    )

    return action, new_agent_state


def _fsm_state_to_action(fsm_state: chex.Array, const: CageConst) -> chex.Array:
    """Map FSM state to Red action index.

    Action encoding for Red (from actions.py):
    - Sleep: 0
    - DiscoverRemoteSystems: 1 + subnet_idx
    - DiscoverNetworkServices: scan_start + host_idx
    - Exploit: exploit_start + exploit_type * num_hosts + host_idx
    - PrivilegeEscalate: privesc_start + host_idx
    - Impact: impact_start + host_idx
    """
    discover_start, scan_start, exploit_start, privesc_start, impact_start = get_red_action_offsets(const)

    def state_0(_):  # DiscoverSubnet(User)
        return discover_start + BLINE_USER_SUBNET

    def state_1(_):  # ScanHost(User1)
        return scan_start + BLINE_USER_HOST

    def state_2(_):  # Exploit(User1)
        return exploit_start + BLINE_EXPLOIT_TYPE * const.num_hosts + BLINE_USER_HOST

    def state_3(_):  # PrivEsc(User1)
        return privesc_start + BLINE_USER_HOST

    def state_4(_):  # DiscoverSubnet(Enterprise)
        return discover_start + BLINE_ENTERPRISE_SUBNET

    def state_5(_):  # ScanHost(Enterprise0)
        return scan_start + BLINE_ENTERPRISE0

    def state_6(_):  # Exploit(Enterprise0)
        return exploit_start + BLINE_EXPLOIT_TYPE * const.num_hosts + BLINE_ENTERPRISE0

    def state_7(_):  # PrivEsc(Enterprise0)
        return privesc_start + BLINE_ENTERPRISE0

    def state_8(_):  # ScanHost(Enterprise2)
        return scan_start + BLINE_ENTERPRISE2

    def state_9(_):  # Exploit(Enterprise2)
        return exploit_start + BLINE_EXPLOIT_TYPE * const.num_hosts + BLINE_ENTERPRISE2

    def state_10(_):  # PrivEsc(Enterprise2)
        return privesc_start + BLINE_ENTERPRISE2

    def state_11(_):  # DiscoverSubnet(Operational)
        return discover_start + BLINE_OPERATIONAL_SUBNET

    def state_12(_):  # ScanHost(Op_Server0)
        return scan_start + BLINE_OP_SERVER0

    def state_13(_):  # Exploit(Op_Server0)
        return exploit_start + BLINE_EXPLOIT_TYPE * const.num_hosts + BLINE_OP_SERVER0

    def state_14(_):  # PrivEsc(Op_Server0)
        return privesc_start + BLINE_OP_SERVER0

    def state_15(_):  # Impact(Op_Server0)
        return impact_start + BLINE_OP_SERVER0

    action = jax.lax.switch(
        fsm_state,
        [state_0, state_1, state_2, state_3, state_4,
         state_5, state_6, state_7, state_8, state_9,
         state_10, state_11, state_12, state_13, state_14, state_15],
        None,
    )

    return jnp.array(action, dtype=jnp.int32)


def bline_get_action_batched(
    agent_states: BLineState,
    red_obs: chex.Array,
    action_masks: chex.Array,
    const: CageConst,
    keys: chex.PRNGKey,
) -> Tuple[chex.Array, BLineState]:
    """Batched version of bline_get_action.

    Args:
        agent_states: Batched BLineState with shape (batch_size,) for each field
        red_obs: Red observations with shape (batch_size, obs_dim)
        action_masks: Action masks with shape (batch_size, num_actions)
        const: Environment constants (shared)
        keys: Random keys with shape (batch_size, 2)

    Returns:
        Tuple of (actions with shape (batch_size,), new_agent_states)
    """
    return jax.vmap(bline_get_action, in_axes=(0, 0, 0, None, 0))(
        agent_states, red_obs, action_masks, const, keys
    )


def bline_reset_batched(batch_size: int) -> BLineState:
    """Create batched initial BLineState.

    Args:
        batch_size: Number of parallel environments

    Returns:
        BLineState with batched arrays
    """
    return BLineState(
        fsm_state=jnp.zeros(batch_size, dtype=jnp.int32),
        last_action_success=jnp.ones(batch_size, dtype=jnp.bool_),
    )
