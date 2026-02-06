"""Differential testing harness for CybORG <-> CAGE-JAX comparison."""

from .action_translator import (
    cyborg_action_to_jax,
    jax_action_to_cyborg,
    get_exploit_class_from_action,
    EXPLOIT_CLASS_TO_JAX_IDX,
)
from .state_comparator import (
    extract_cyborg_state,
    extract_jax_state,
    compare_states,
    compare_reward_deltas,
    compute_reward_deltas,
    get_compromised_hosts,
    get_privileged_hosts,
    StateDiff,
    StateSnapshot,
)
from .harness import (
    DifferentialHarness,
    JaxOnlyHarness,
    TestResult,
    StepResult,
    is_cyborg_available,
    sleep_policy,
    monitor_policy,
    reactive_remove_policy,
    reactive_restore_policy,
    scripted_red_policy_factory,
    scripted_blue_policy_factory,
    conditional_blue_policy_factory,
)
