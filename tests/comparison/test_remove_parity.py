"""Differential tests comparing CybORG Remove behavior to JAX.

Every test runs both CybORG and JAX with identical actions and compares results.
These tests verify reward parity for the Remove action.
"""

import pytest
from tests.differential.harness import DifferentialHarness, is_cyborg_available
from tests.differential.state_comparator import StateSnapshot, COMPROMISE_USER, COMPROMISE_PRIVILEGED
from tests.comparison.policies import (
    scripted_blue_policy_factory,
    scripted_red_policy_factory,
    sleep_policy,
    monitor_policy,
)
from tests.comparison.scenarios import (
    red_discover_subnet,
    red_scan_host,
    red_exploit_host,
    red_privesc_host,
    blue_remove_host,
    blue_analyse_host,
    SUBNET_USER,
    EXPLOIT_SSH,
)
from jaxmarl.environments.cage.actions import BLUE_SLEEP, BLUE_MONITOR


requires_cyborg = pytest.mark.skipif(
    not is_cyborg_available(),
    reason="CybORG not installed"
)


@requires_cyborg
class TestRemovePenaltyCybORGParity:
    """Differential tests comparing CybORG Remove behavior to JAX.

    Key CybORG behavior:
    - Remove succeeds if there's ANY Red session on the host
    - Remove removes user-level sessions (not root/SYSTEM)
    - Remove does NOT have an action penalty (unlike Restore)
    - Remove success is NOT gated by activity detection or privilege level
    """

    def test_remove_on_user_compromise_no_penalty(self):
        """CybORG vs JAX: Remove on USER-level compromised host should match rewards."""
        harness = DifferentialHarness(seed=42, max_steps=10, verbose=False)

        red_actions = [
            red_discover_subnet(SUBNET_USER),
            red_scan_host('User1'),
            red_exploit_host('User1', EXPLOIT_SSH),
            0,  # Sleep - stay at USER level
        ]

        blue_actions = [
            BLUE_MONITOR,
            BLUE_MONITOR,
            BLUE_MONITOR,
            blue_remove_host('User1'),  # Remove at step 3 when Red has USER access
        ]

        red_policy = scripted_red_policy_factory(red_actions)
        blue_policy = scripted_blue_policy_factory(blue_actions)

        result = harness.run_episode(blue_policy, red_policy)

        remove_step = result.step_results[3]  # Step 4 (0-indexed as 3)

        assert abs(remove_step.cyborg_state.reward_blue - remove_step.jax_state.reward_blue) < 0.02, \
            f"Blue reward mismatch at Remove step: CybORG={remove_step.cyborg_state.reward_blue}, JAX={remove_step.jax_state.reward_blue}"

        assert remove_step.jax_state.red_privilege.get('User1', 0) == 0, \
            f"Red should have no privilege on User1 after Remove"

    def test_remove_on_privileged_compromise_no_extra_penalty(self):
        """CybORG vs JAX: Remove on PRIVILEGED host - verify no extra penalty.

        This is the key test case from the investigation. CybORG Remove doesn't
        apply extra penalties based on privilege level. Remove simply attempts
        to kill user-level sessions.
        """
        harness = DifferentialHarness(seed=42, max_steps=10, verbose=False)

        red_actions = [
            red_discover_subnet(SUBNET_USER),
            red_scan_host('User1'),
            red_exploit_host('User1', EXPLOIT_SSH),
            red_privesc_host('User1'),
            0,  # Sleep
        ]

        blue_actions = [
            BLUE_MONITOR,
            BLUE_MONITOR,
            BLUE_MONITOR,
            BLUE_MONITOR,
            blue_remove_host('User1'),
        ]

        red_policy = scripted_red_policy_factory(red_actions)
        blue_policy = scripted_blue_policy_factory(blue_actions)

        result = harness.run_episode(blue_policy, red_policy)

        remove_step = result.step_results[4]

        assert abs(remove_step.cyborg_state.reward_blue - remove_step.jax_state.reward_blue) < 0.02, \
            f"Blue reward mismatch at Remove step: CybORG={remove_step.cyborg_state.reward_blue}, JAX={remove_step.jax_state.reward_blue}"

    def test_remove_on_clean_host_no_penalty(self):
        """CybORG vs JAX: Remove on non-compromised host should not penalize."""
        harness = DifferentialHarness(seed=42, max_steps=5, verbose=False)

        red_actions = [
            0,  # Sleep
            0,  # Sleep
        ]

        blue_actions = [
            BLUE_MONITOR,
            blue_remove_host('Enterprise0'),  # Remove on clean host
        ]

        red_policy = scripted_red_policy_factory(red_actions)
        blue_policy = scripted_blue_policy_factory(blue_actions)

        result = harness.run_episode(blue_policy, red_policy)

        remove_step = result.step_results[1]  # Step 2 (0-indexed as 1)

        assert abs(remove_step.cyborg_state.reward_blue - remove_step.jax_state.reward_blue) < 0.02, \
            f"Blue reward mismatch on clean host Remove: CybORG={remove_step.cyborg_state.reward_blue}, JAX={remove_step.jax_state.reward_blue}"

    def test_remove_vs_red_privesc_race(self):
        """CybORG vs JAX: Remove and PrivEsc happening simultaneously."""
        harness = DifferentialHarness(seed=42, max_steps=10, verbose=False)

        red_actions = [
            red_discover_subnet(SUBNET_USER),
            red_scan_host('User2'),
            red_exploit_host('User2', EXPLOIT_SSH),
            red_privesc_host('User2'),
            0,
        ]

        blue_actions = [
            BLUE_MONITOR,
            BLUE_MONITOR,
            BLUE_MONITOR,
            blue_remove_host('User2'),
            BLUE_MONITOR,
        ]

        red_policy = scripted_red_policy_factory(red_actions)
        blue_policy = scripted_blue_policy_factory(blue_actions)

        result = harness.run_episode(blue_policy, red_policy)

        for sr in result.step_results:
            cyborg_blue = sr.cyborg_state.reward_blue
            jax_blue = sr.jax_state.reward_blue
            assert abs(cyborg_blue - jax_blue) < 0.02, \
                f"Step {sr.step}: Blue reward mismatch CybORG={cyborg_blue}, JAX={jax_blue}"

    def test_remove_after_analyse_detects_activity(self):
        """CybORG vs JAX: Remove after Analyse has detected activity."""
        harness = DifferentialHarness(seed=42, max_steps=10, verbose=False)

        red_actions = [
            red_discover_subnet(SUBNET_USER),
            red_scan_host('User2'),
            red_exploit_host('User2', EXPLOIT_SSH),
            0,  # Sleep
            0,  # Sleep
        ]

        blue_actions = [
            BLUE_MONITOR,
            BLUE_MONITOR,
            BLUE_MONITOR,
            blue_analyse_host('User2'),  # Analyse to detect
            blue_remove_host('User2'),   # Then remove
        ]

        red_policy = scripted_red_policy_factory(red_actions)
        blue_policy = scripted_blue_policy_factory(blue_actions)

        result = harness.run_episode(blue_policy, red_policy)

        for sr in result.step_results:
            cyborg_blue = sr.cyborg_state.reward_blue
            jax_blue = sr.jax_state.reward_blue
            assert abs(cyborg_blue - jax_blue) < 0.02, \
                f"Step {sr.step}: Blue reward mismatch CybORG={cyborg_blue}, JAX={jax_blue}"

    @pytest.mark.parametrize("host", ['User1', 'User2', 'User3', 'User4'])
    def test_remove_on_different_user_hosts(self, host):
        """CybORG vs JAX: Remove behavior should match across different hosts."""
        harness = DifferentialHarness(seed=42, max_steps=8, verbose=False)

        red_actions = [
            red_discover_subnet(SUBNET_USER),
            red_scan_host(host),
            red_exploit_host(host, EXPLOIT_SSH),
            0,  # Sleep
        ]

        blue_actions = [
            BLUE_MONITOR,
            BLUE_MONITOR,
            BLUE_MONITOR,
            blue_remove_host(host),
        ]

        red_policy = scripted_red_policy_factory(red_actions)
        blue_policy = scripted_blue_policy_factory(blue_actions)

        result = harness.run_episode(blue_policy, red_policy)

        remove_step = result.step_results[3]
        assert abs(remove_step.cyborg_state.reward_blue - remove_step.jax_state.reward_blue) < 0.02, \
            f"Remove on {host}: Blue reward mismatch CybORG={remove_step.cyborg_state.reward_blue}, JAX={remove_step.jax_state.reward_blue}"

    def test_multiple_removes_in_episode(self):
        """CybORG vs JAX: Multiple Remove actions in one episode."""
        harness = DifferentialHarness(seed=42, max_steps=12, verbose=False)

        red_actions = [
            red_discover_subnet(SUBNET_USER),
            red_scan_host('User1'),
            red_exploit_host('User1', EXPLOIT_SSH),
            red_scan_host('User2'),
            red_exploit_host('User2', EXPLOIT_SSH),
            0,
            0,
        ]

        blue_actions = [
            BLUE_MONITOR,
            BLUE_MONITOR,
            BLUE_MONITOR,
            BLUE_MONITOR,
            BLUE_MONITOR,
            blue_remove_host('User1'),
            blue_remove_host('User2'),
        ]

        red_policy = scripted_red_policy_factory(red_actions)
        blue_policy = scripted_blue_policy_factory(blue_actions)

        result = harness.run_episode(blue_policy, red_policy)

        for sr in result.step_results:
            cyborg_blue = sr.cyborg_state.reward_blue
            jax_blue = sr.jax_state.reward_blue
            assert abs(cyborg_blue - jax_blue) < 0.02, \
                f"Step {sr.step}: Blue reward mismatch CybORG={cyborg_blue}, JAX={jax_blue}"
