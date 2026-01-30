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


# B_lineAgent target hosts are now looked up dynamically from CageConst
# to support scaled scenarios where host indices differ from Scenario2.
# The target host names (User1, Enterprise1, Enterprise2, Op_Server0) are
# consistent across scenarios, but their indices vary based on alphabetical
# ordering of all hosts in the scenario.

# Jump-back table: on failure at state i, jump to state BLINE_JUMP_BACK[i]
# FSM has 15 states (0-14) matching CybORG's B_lineAgent exactly
BLINE_JUMP_BACK = jnp.array([
    0,   # State 0:  DiscoverRemoteSystems(User subnet) -> retry
    1,   # State 1:  DiscoverNetworkServices(User1) -> retry
    2,   # State 2:  Exploit(User1) -> retry
    2,   # State 3:  PrivEsc(User1) -> back to exploit
    2,   # State 4:  DiscoverNetworkServices(Enterprise1) -> back to User exploit
    2,   # State 5:  Exploit(Enterprise1) -> back to User exploit
    5,   # State 6:  PrivEsc(Enterprise1) -> back to Enterprise1 exploit
    5,   # State 7:  DiscoverRemoteSystems(Enterprise subnet) -> back to Enterprise1 exploit
    5,   # State 8:  DiscoverNetworkServices(Enterprise2) -> back to Enterprise1 exploit
    5,   # State 9:  Exploit(Enterprise2) -> back to Enterprise1 exploit
    9,   # State 10: PrivEsc(Enterprise2) -> back to Enterprise2 exploit
    9,   # State 11: DiscoverNetworkServices(Op_Server0) -> back to Enterprise2 exploit
    9,   # State 12: Exploit(Op_Server0) -> back to Enterprise2 exploit
    12,  # State 13: PrivEsc(Op_Server0) -> back to Op_Server0 exploit
    13,  # State 14: Impact(Op_Server0) -> back to Op_Server0 privesc
])

# Default exploit type for B_lineAgent (SSH brute force = 0)
# NOTE: This is only used as a fallback. The actual exploit selection is now
# computed dynamically based on each host's available services to match
# CybORG's ExploitRemoteService behavior.
BLINE_EXPLOIT_TYPE = 0


def _get_host_first_exploit(host_idx: chex.Array, const: CageConst) -> chex.Array:
    """Get the first available exploit for a host based on its initial services.

    CybORG's ExploitRemoteService selects an exploit that matches the target's
    services. This function computes the first valid exploit for a given host.
    """
    services = const.initial_services[host_idx]  # (num_services,)

    # For each exploit, check if the host has a service that's vulnerable to it
    def check_exploit(exploit_idx):
        # Check all services to see if any is vulnerable to this exploit
        has_service = jnp.any(services & const.service_exploits[:, exploit_idx])
        return has_service

    # Check exploits in order and find first valid one
    exploit_available = jax.vmap(check_exploit)(jnp.arange(const.num_exploits))
    # Return first available exploit (default to 0 if none found)
    first_exploit = jnp.argmax(exploit_available)
    return first_exploit

# Max FSM state (0-14, 15 total states)
BLINE_MAX_STATE = 14


@struct.dataclass
class BLineState:
    """State for B_lineAgent finite state machine."""
    fsm_state: chex.Array  # Current FSM state (0-14)
    last_action_success: chex.Array  # Track last action result
    target_user_idx: chex.Array  # Which User host to attack (index into bline_user_hosts)


def bline_reset(key: chex.PRNGKey = None) -> BLineState:
    """Reset B_lineAgent state to initial FSM state 0.

    Args:
        key: Random key for selecting target User host. If None, defaults to User1 (idx 0).
             CybORG's B_lineAgent randomly selects which User host to attack.
    """
    if key is None:
        target_user_idx = jnp.array(0, dtype=jnp.int32)
    else:
        target_user_idx = jax.random.randint(key, (), 0, 4)
    return BLineState(
        fsm_state=jnp.array(0, dtype=jnp.int32),
        last_action_success=jnp.array(True, dtype=jnp.bool_),
        target_user_idx=target_user_idx,
    )


def bline_get_action(
    agent_state: BLineState,
    red_obs: chex.Array,
    action_mask: chex.Array,
    const: CageConst,
    key: chex.PRNGKey,
) -> Tuple[chex.Array, BLineState]:
    """Get B_lineAgent action based on FSM state.

    The B_lineAgent follows a deterministic 15-state FSM matching CybORG:
    - State 0:  DiscoverRemoteSystems(User subnet)
    - State 1:  DiscoverNetworkServices(User1)
    - State 2:  Exploit(User1)
    - State 3:  PrivEsc(User1)
    - State 4:  DiscoverNetworkServices(Enterprise1)
    - State 5:  Exploit(Enterprise1)
    - State 6:  PrivEsc(Enterprise1)
    - State 7:  DiscoverRemoteSystems(Enterprise subnet)
    - State 8:  DiscoverNetworkServices(Enterprise2)
    - State 9:  Exploit(Enterprise2)
    - State 10: PrivEsc(Enterprise2)
    - State 11: DiscoverNetworkServices(Op_Server0)
    - State 12: Exploit(Op_Server0)
    - State 13: PrivEsc(Op_Server0)
    - State 14: Impact(Op_Server0)

    On action failure, jumps back using BLINE_JUMP_BACK table.
    On success, advances to next state (capped at 14).

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

    # Get action for the new FSM state (pass agent_state for random user target)
    # CybORG's B_lineAgent doesn't check action masks - it just tries actions
    # and they succeed/fail based on game rules
    action = _fsm_state_to_action(new_fsm_state, const, agent_state.target_user_idx)

    new_agent_state = BLineState(
        fsm_state=new_fsm_state,
        last_action_success=last_success,
        target_user_idx=agent_state.target_user_idx,  # Preserve random target
    )

    return action, new_agent_state


def _fsm_state_to_action(fsm_state: chex.Array, const: CageConst, target_user_idx: chex.Array) -> chex.Array:
    """Map FSM state to Red action index.

    Action encoding for Red (from actions.py):
    - Sleep: 0
    - DiscoverRemoteSystems: 1 + subnet_idx
    - DiscoverNetworkServices: scan_start + host_idx
    - Exploit: exploit_start + host_idx * num_exploits + exploit_type
    - PrivilegeEscalate: privesc_start + host_idx
    - Impact: impact_start + host_idx

    15-state FSM matching CybORG's B_lineAgent exactly.
    Host and subnet indices are looked up dynamically from const for scenario compatibility.

    CybORG's B_lineAgent randomly selects which User host to attack. The target_user_idx
    selects from bline_user_hosts (User1-4), and the connected Enterprise is looked up
    from user_to_enterprise mapping.
    """
    discover_start, scan_start, exploit_start, privesc_start, impact_start = get_red_action_offsets(const)

    # Get random target User host and its connected Enterprise
    # target_user_idx selects from bline_user_hosts (0=User1, 1=User2, 2=User3, 3=User4)
    user_host = const.bline_user_hosts[target_user_idx]
    enterprise1 = const.user_to_enterprise[target_user_idx]  # Connected Enterprise for this User
    enterprise2 = const.bline_enterprise2
    op_server0 = const.bline_op_server0

    # Get subnet indices dynamically from host_subnet
    user_subnet = const.host_subnet[user_host]
    enterprise_subnet = const.host_subnet[enterprise1]

    def state_0(_):  # DiscoverRemoteSystems(User subnet)
        return discover_start + user_subnet

    def state_1(_):  # DiscoverNetworkServices(User1)
        return scan_start + user_host

    def state_2(_):  # Exploit(User1)
        exploit_type = _get_host_first_exploit(user_host, const)
        return exploit_start + user_host * const.num_exploits + exploit_type

    def state_3(_):  # PrivEsc(User1)
        return privesc_start + user_host

    def state_4(_):  # DiscoverNetworkServices(Enterprise1)
        return scan_start + enterprise1

    def state_5(_):  # Exploit(Enterprise1)
        exploit_type = _get_host_first_exploit(enterprise1, const)
        return exploit_start + enterprise1 * const.num_exploits + exploit_type

    def state_6(_):  # PrivEsc(Enterprise1)
        return privesc_start + enterprise1

    def state_7(_):  # DiscoverRemoteSystems(Enterprise subnet)
        return discover_start + enterprise_subnet

    def state_8(_):  # DiscoverNetworkServices(Enterprise2)
        return scan_start + enterprise2

    def state_9(_):  # Exploit(Enterprise2)
        exploit_type = _get_host_first_exploit(enterprise2, const)
        return exploit_start + enterprise2 * const.num_exploits + exploit_type

    def state_10(_):  # PrivEsc(Enterprise2)
        return privesc_start + enterprise2

    def state_11(_):  # DiscoverNetworkServices(Op_Server0)
        return scan_start + op_server0

    def state_12(_):  # Exploit(Op_Server0)
        exploit_type = _get_host_first_exploit(op_server0, const)
        return exploit_start + op_server0 * const.num_exploits + exploit_type

    def state_13(_):  # PrivEsc(Op_Server0)
        return privesc_start + op_server0

    def state_14(_):  # Impact(Op_Server0)
        return impact_start + op_server0

    action = jax.lax.switch(
        fsm_state,
        [state_0, state_1, state_2, state_3, state_4,
         state_5, state_6, state_7, state_8, state_9,
         state_10, state_11, state_12, state_13, state_14],
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


def bline_reset_batched(batch_size: int, key: chex.PRNGKey = None) -> BLineState:
    """Create batched initial BLineState with random User targets.

    Args:
        batch_size: Number of parallel environments
        key: Random key for selecting target User hosts. If None, all target User1.

    Returns:
        BLineState with batched arrays
    """
    if key is None:
        target_user_idx = jnp.zeros(batch_size, dtype=jnp.int32)
    else:
        target_user_idx = jax.random.randint(key, (batch_size,), 0, 4)
    return BLineState(
        fsm_state=jnp.zeros(batch_size, dtype=jnp.int32),
        last_action_success=jnp.ones(batch_size, dtype=jnp.bool_),
        target_user_idx=target_user_idx,
    )


# ============================================================================
# RedMeanderAgent - Exploratory/Opportunistic Red Agent
# ============================================================================
# Unlike B_lineAgent's fixed 15-state FSM, RedMeander explores the network
# opportunistically: scans all discovered hosts, exploits available targets,
# and escalates privileges where possible. The order of targets is randomized.


@struct.dataclass
class MeanderState:
    """State for RedMeanderAgent exploratory behavior."""
    scanned_subnets: chex.Array      # (num_subnets,) bool - subnets we've run DiscoverRemoteSystems on
    scanned_ips: chex.Array          # (num_hosts,) bool - hosts we've run DiscoverNetworkServices on
    exploited_ips: chex.Array        # (num_hosts,) bool - hosts we've successfully exploited
    escalated_hosts: chex.Array      # (num_hosts,) bool - hosts where we've escalated privileges
    host_ip_known: chex.Array        # (num_hosts,) bool - hosts where we learned IP→host mapping
    last_host_idx: chex.Array        # scalar int (-1 = none) - last host we tried PrivEsc on
    last_ip_idx: chex.Array          # scalar int (-1 = none) - last host we tried Exploit on
    last_action_success: chex.Array  # scalar bool - result of last action


def meander_reset(const: CageConst, key: chex.PRNGKey = None) -> MeanderState:
    """Reset MeanderState to initial state.

    Args:
        const: Environment constants for array sizing
        key: Random key (unused, for API compatibility)

    Returns:
        Fresh MeanderState with all tracking arrays zeroed
    """
    return MeanderState(
        scanned_subnets=jnp.zeros(const.num_subnets, dtype=jnp.bool_),
        scanned_ips=jnp.zeros(const.num_hosts, dtype=jnp.bool_),
        exploited_ips=jnp.zeros(const.num_hosts, dtype=jnp.bool_),
        escalated_hosts=jnp.zeros(const.num_hosts, dtype=jnp.bool_),
        host_ip_known=jnp.zeros(const.num_hosts, dtype=jnp.bool_),
        last_host_idx=jnp.array(-1, dtype=jnp.int32),
        last_ip_idx=jnp.array(-1, dtype=jnp.int32),
        last_action_success=jnp.array(True, dtype=jnp.bool_),
    )


def meander_get_action(
    agent_state: MeanderState,
    red_obs: chex.Array,
    action_mask: chex.Array,
    const: CageConst,
    key: chex.PRNGKey,
) -> Tuple[chex.Array, MeanderState]:
    """Get RedMeanderAgent action based on opportunistic exploration.

    Priority order (from CybORG's Meander.py):
    1. Impact Op_Server0 if escalated there
    2. DiscoverSubnet for unscanned subnets (random order)
    3. DiscoverNetworkServices on discovered but unscanned hosts (random order)
    4. PrivilegeEscalate on exploited but not escalated hosts (random order)
    5. ExploitRemoteService on scanned but not exploited hosts (random order)

    On failure, we backtrack by removing the failed host from our tracking.

    Args:
        agent_state: Current MeanderState
        red_obs: Red agent observation (first element is last_action_success)
        action_mask: Valid action mask
        const: Environment constants
        key: Random key for shuffling targets

    Returns:
        Tuple of (action_index, new_agent_state)
    """
    last_success = red_obs[0] > 0.5

    agent_state = _meander_process_result(agent_state, last_success, const)

    discover_start, scan_start, exploit_start, privesc_start, impact_start = get_red_action_offsets(const)

    key, k1, k2, k3, k4 = jax.random.split(key, 5)

    op_server0 = const.bline_op_server0
    can_impact = agent_state.escalated_hosts[op_server0]
    impact_action = impact_start + op_server0

    def do_impact(_):
        return impact_action, agent_state

    def try_discover_or_continue(_):
        subnet_perm = jax.random.permutation(k1, const.num_subnets)

        def find_unscanned_subnet(carry, subnet_idx):
            found, action, state = carry
            subnet = subnet_perm[subnet_idx]
            is_unscanned = ~state.scanned_subnets[subnet]
            is_valid = action_mask[discover_start + subnet]
            should_use = ~found & is_unscanned & is_valid

            new_state = jax.lax.cond(
                should_use,
                lambda s: s.replace(scanned_subnets=s.scanned_subnets.at[subnet].set(True)),
                lambda s: s,
                state,
            )
            new_action = jnp.where(should_use, discover_start + subnet, action)
            new_found = found | should_use

            return (new_found, new_action, new_state), None

        (found_discover, discover_action, state_after_discover), _ = jax.lax.scan(
            find_unscanned_subnet,
            (False, jnp.array(0, dtype=jnp.int32), agent_state),
            jnp.arange(const.num_subnets),
        )

        def do_discover(_):
            return discover_action, state_after_discover

        def try_scan_or_continue(_):
            host_perm = jax.random.permutation(k2, const.num_hosts)

            def find_unscannable_host(carry, host_idx):
                found, action, state = carry
                host = host_perm[host_idx]
                is_unscanned = ~state.scanned_ips[host]
                is_valid = action_mask[scan_start + host]
                should_use = ~found & is_unscanned & is_valid

                new_state = jax.lax.cond(
                    should_use,
                    lambda s: s.replace(scanned_ips=s.scanned_ips.at[host].set(True)),
                    lambda s: s,
                    state,
                )
                new_action = jnp.where(should_use, scan_start + host, action)
                new_found = found | should_use

                return (new_found, new_action, new_state), None

            (found_scan, scan_action, state_after_scan), _ = jax.lax.scan(
                find_unscannable_host,
                (False, jnp.array(0, dtype=jnp.int32), state_after_discover),
                jnp.arange(const.num_hosts),
            )

            def do_scan(_):
                return scan_action, state_after_scan

            def try_privesc_or_continue(_):
                host_perm_privesc = jax.random.permutation(k3, const.num_hosts)

                def find_privesc_host(carry, host_idx):
                    found, action, state = carry
                    host = host_perm_privesc[host_idx]
                    is_exploited = state.exploited_ips[host]
                    is_not_escalated = ~state.escalated_hosts[host]
                    is_valid = action_mask[privesc_start + host]
                    should_use = ~found & is_exploited & is_not_escalated & is_valid

                    new_state = jax.lax.cond(
                        should_use,
                        lambda s: s.replace(
                            escalated_hosts=s.escalated_hosts.at[host].set(True),
                            last_host_idx=host,
                        ),
                        lambda s: s,
                        state,
                    )
                    new_action = jnp.where(should_use, privesc_start + host, action)
                    new_found = found | should_use

                    return (new_found, new_action, new_state), None

                (found_privesc, privesc_action, state_after_privesc), _ = jax.lax.scan(
                    find_privesc_host,
                    (False, jnp.array(0, dtype=jnp.int32), state_after_scan),
                    jnp.arange(const.num_hosts),
                )

                def do_privesc(_):
                    return privesc_action, state_after_privesc

                def try_exploit_or_sleep(_):
                    host_perm_exploit = jax.random.permutation(k4, const.num_hosts)

                    def find_exploit_host(carry, host_idx):
                        found, action, state = carry
                        host = host_perm_exploit[host_idx]
                        is_scanned = state.scanned_ips[host]
                        is_not_exploited = ~state.exploited_ips[host]

                        exploit_type = _get_host_first_exploit(host, const)
                        exploit_action_idx = exploit_start + host * const.num_exploits + exploit_type
                        is_valid = action_mask[exploit_action_idx]

                        should_use = ~found & is_scanned & is_not_exploited & is_valid

                        new_state = jax.lax.cond(
                            should_use,
                            lambda s: s.replace(
                                exploited_ips=s.exploited_ips.at[host].set(True),
                                last_ip_idx=host,
                            ),
                            lambda s: s,
                            state,
                        )
                        new_action = jnp.where(should_use, exploit_action_idx, action)
                        new_found = found | should_use

                        return (new_found, new_action, new_state), None

                    (found_exploit, exploit_action, state_after_exploit), _ = jax.lax.scan(
                        find_exploit_host,
                        (False, jnp.array(0, dtype=jnp.int32), state_after_privesc),
                        jnp.arange(const.num_hosts),
                    )

                    def do_exploit(_):
                        return exploit_action, state_after_exploit

                    def do_sleep(_):
                        return jnp.array(0, dtype=jnp.int32), state_after_privesc

                    return jax.lax.cond(found_exploit, do_exploit, do_sleep, None)

                return jax.lax.cond(found_privesc, do_privesc, try_exploit_or_sleep, None)

            return jax.lax.cond(found_scan, do_scan, try_privesc_or_continue, None)

        return jax.lax.cond(found_discover, do_discover, try_scan_or_continue, None)

    action, new_state = jax.lax.cond(can_impact, do_impact, try_discover_or_continue, None)

    action = jax.lax.cond(
        action_mask[action],
        lambda: action,
        lambda: jnp.array(0, dtype=jnp.int32),
    )

    new_state = new_state.replace(last_action_success=last_success)

    return action, new_state


def _meander_process_result(
    state: MeanderState,
    success: chex.Array,
    const: CageConst,
) -> MeanderState:
    """Process the result of the last action and update state accordingly.

    On exploit failure: remove from exploited_ips, cascade remove Op/Enterprise hosts
    On privesc failure: remove from escalated_hosts AND exploited_ips

    Args:
        state: Current MeanderState
        success: Whether last action succeeded
        const: Environment constants

    Returns:
        Updated MeanderState with failed hosts removed from tracking
    """
    had_last_ip = state.last_ip_idx >= 0
    had_last_host = state.last_host_idx >= 0

    def process_exploit_result(s):
        def on_exploit_fail(s):
            ip_idx = s.last_ip_idx
            new_exploited = s.exploited_ips.at[ip_idx].set(False)

            def is_op_host(host_idx):
                return (host_idx >= 4) & (host_idx <= 7)

            def is_ent_host(host_idx):
                return (host_idx >= 1) & (host_idx <= 3)

            has_op = jnp.any(s.escalated_hosts & jax.vmap(is_op_host)(jnp.arange(const.num_hosts)))

            def remove_op_hosts(new_exploited, new_escalated):
                for i in range(4, 8):
                    should_remove = s.escalated_hosts[i]
                    new_escalated = jnp.where(should_remove, new_escalated.at[i].set(False), new_escalated)
                    new_exploited = jnp.where(should_remove, new_exploited.at[i].set(False), new_exploited)
                return new_exploited, new_escalated

            def remove_ent_hosts(new_exploited, new_escalated):
                has_ent = jnp.any(s.escalated_hosts & jax.vmap(is_ent_host)(jnp.arange(const.num_hosts)))
                for i in range(1, 4):
                    should_remove = s.escalated_hosts[i] & has_ent
                    new_escalated = jnp.where(should_remove, new_escalated.at[i].set(False), new_escalated)
                    new_exploited = jnp.where(should_remove, new_exploited.at[i].set(False), new_exploited)
                return new_exploited, new_escalated

            new_escalated = s.escalated_hosts

            new_exploited, new_escalated = jax.lax.cond(
                has_op,
                lambda args: remove_op_hosts(*args),
                lambda args: remove_ent_hosts(*args),
                (new_exploited, new_escalated),
            )

            return s.replace(
                exploited_ips=new_exploited,
                escalated_hosts=new_escalated,
                last_ip_idx=jnp.array(-1, dtype=jnp.int32),
            )

        def on_exploit_success(s):
            return s.replace(
                host_ip_known=s.host_ip_known.at[s.last_ip_idx].set(True),
                last_ip_idx=jnp.array(-1, dtype=jnp.int32),
            )

        return jax.lax.cond(success, on_exploit_success, on_exploit_fail, s)

    def no_exploit_processing(s):
        return s

    state = jax.lax.cond(had_last_ip, process_exploit_result, no_exploit_processing, state)

    def process_privesc_result(s):
        def on_privesc_fail(s):
            host_idx = s.last_host_idx
            new_escalated = s.escalated_hosts.at[host_idx].set(False)
            new_exploited = s.exploited_ips.at[host_idx].set(False)
            return s.replace(
                escalated_hosts=new_escalated,
                exploited_ips=new_exploited,
                last_host_idx=jnp.array(-1, dtype=jnp.int32),
            )

        def on_privesc_success(s):
            return s.replace(last_host_idx=jnp.array(-1, dtype=jnp.int32))

        return jax.lax.cond(success, on_privesc_success, on_privesc_fail, s)

    def no_privesc_processing(s):
        return s

    state = jax.lax.cond(had_last_host, process_privesc_result, no_privesc_processing, state)

    return state


def meander_get_action_batched(
    agent_states: MeanderState,
    red_obs: chex.Array,
    action_masks: chex.Array,
    const: CageConst,
    keys: chex.PRNGKey,
) -> Tuple[chex.Array, MeanderState]:
    """Batched version of meander_get_action.

    Args:
        agent_states: Batched MeanderState with shape (batch_size,) for each field
        red_obs: Red observations with shape (batch_size, obs_dim)
        action_masks: Action masks with shape (batch_size, num_actions)
        const: Environment constants (shared)
        keys: Random keys with shape (batch_size, 2)

    Returns:
        Tuple of (actions with shape (batch_size,), new_agent_states)
    """
    return jax.vmap(meander_get_action, in_axes=(0, 0, 0, None, 0))(
        agent_states, red_obs, action_masks, const, keys
    )


def meander_reset_batched(batch_size: int, const: CageConst, key: chex.PRNGKey = None) -> MeanderState:
    """Create batched initial MeanderState.

    Args:
        batch_size: Number of parallel environments
        const: Environment constants for array sizing
        key: Random key (unused, for API compatibility)

    Returns:
        MeanderState with batched arrays
    """
    return MeanderState(
        scanned_subnets=jnp.zeros((batch_size, const.num_subnets), dtype=jnp.bool_),
        scanned_ips=jnp.zeros((batch_size, const.num_hosts), dtype=jnp.bool_),
        exploited_ips=jnp.zeros((batch_size, const.num_hosts), dtype=jnp.bool_),
        escalated_hosts=jnp.zeros((batch_size, const.num_hosts), dtype=jnp.bool_),
        host_ip_known=jnp.zeros((batch_size, const.num_hosts), dtype=jnp.bool_),
        last_host_idx=jnp.full(batch_size, -1, dtype=jnp.int32),
        last_ip_idx=jnp.full(batch_size, -1, dtype=jnp.int32),
        last_action_success=jnp.ones(batch_size, dtype=jnp.bool_),
    )
