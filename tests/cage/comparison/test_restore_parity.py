"""Differential tests comparing CybORG Restore behavior to JAX.

Every test runs both CybORG and JAX with identical actions and compares results.
These tests verify reward and state parity for the Restore action.
"""

import pytest
from tests.cage.differential.harness import DifferentialHarness, is_cyborg_available
from tests.cage.differential.state_comparator import COMPROMISE_USER
from tests.cage.comparison.policies import (
    scripted_blue_policy_factory,
    scripted_red_policy_factory,
)
from tests.cage.comparison.scenarios import (
    red_discover_subnet,
    red_scan_host,
    red_exploit_host,
    red_privesc_host,
    blue_restore_host,
    blue_analyse_host,
    SUBNET_USER,
    EXPLOIT_SSH,
)
from jaxmarl.environments.cage.actions import BLUE_MONITOR


requires_cyborg = pytest.mark.skipif(
    not is_cyborg_available(),
    reason="CybORG not installed"
)


@requires_cyborg
class TestRestoreCybORGParity:
    """Differential tests comparing CybORG Restore behavior to JAX.

    Key CybORG behavior:
    - Restore has an action cost of -1.0 (unlike Remove which has 0 cost)
    - Restore completely resets the host to initial state
    - Restore removes ALL Red sessions on the host (user and root)
    - Restore removes decoys deployed on the host
    """

    def test_restore_action_cost(self):
        """CybORG vs JAX: Restore should have -1.0 action cost."""
        harness = DifferentialHarness(seed=42, max_steps=10, verbose=False)

        red_actions = [
            red_discover_subnet(SUBNET_USER),
            red_scan_host('User1'),
            red_exploit_host('User1', EXPLOIT_SSH),
            0,  # Sleep
        ]

        blue_actions = [
            BLUE_MONITOR,
            BLUE_MONITOR,
            BLUE_MONITOR,
            blue_restore_host('User1'),
        ]

        red_policy = scripted_red_policy_factory(red_actions)
        blue_policy = scripted_blue_policy_factory(blue_actions)

        result = harness.run_episode(blue_policy, red_policy)

        restore_step = result.step_results[3]

        assert abs(restore_step.cyborg_state.reward_blue - restore_step.jax_state.reward_blue) < 0.02, \
            f"Blue reward mismatch at Restore step: CybORG={restore_step.cyborg_state.reward_blue}, JAX={restore_step.jax_state.reward_blue}"

    def test_restore_on_user_compromise(self):
        """CybORG vs JAX: Restore on USER-level compromised host."""
        harness = DifferentialHarness(seed=42, max_steps=10, verbose=False)

        red_actions = [
            red_discover_subnet(SUBNET_USER),
            red_scan_host('User2'),
            red_exploit_host('User2', EXPLOIT_SSH),
            0,  # Sleep - stay at USER level
            0,  # Sleep
        ]

        blue_actions = [
            BLUE_MONITOR,
            BLUE_MONITOR,
            BLUE_MONITOR,
            BLUE_MONITOR,
            blue_restore_host('User2'),
        ]

        red_policy = scripted_red_policy_factory(red_actions)
        blue_policy = scripted_blue_policy_factory(blue_actions)

        result = harness.run_episode(blue_policy, red_policy)

        restore_step = result.step_results[4]

        assert abs(restore_step.cyborg_state.reward_blue - restore_step.jax_state.reward_blue) < 0.02, \
            f"Blue reward mismatch: CybORG={restore_step.cyborg_state.reward_blue}, JAX={restore_step.jax_state.reward_blue}"

        assert restore_step.jax_state.red_privilege.get('User2', 0) == 0, \
            "Red should have no privilege on User2 after Restore"

    def test_restore_on_privileged_compromise(self):
        """CybORG vs JAX: Restore on PRIVILEGED host removes root access."""
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
            blue_restore_host('User1'),
        ]

        red_policy = scripted_red_policy_factory(red_actions)
        blue_policy = scripted_blue_policy_factory(blue_actions)

        result = harness.run_episode(blue_policy, red_policy)

        restore_step = result.step_results[4]

        assert abs(restore_step.cyborg_state.reward_blue - restore_step.jax_state.reward_blue) < 0.02, \
            f"Blue reward mismatch: CybORG={restore_step.cyborg_state.reward_blue}, JAX={restore_step.jax_state.reward_blue}"

        assert restore_step.jax_state.red_privilege.get('User1', 0) == 0, \
            "Red should have no privilege on User1 after Restore"

    def test_restore_on_clean_host(self):
        """CybORG vs JAX: Restore on non-compromised host still costs -1.0."""
        harness = DifferentialHarness(seed=42, max_steps=5, verbose=False)

        red_actions = [
            0,  # Sleep
            0,  # Sleep
        ]

        blue_actions = [
            BLUE_MONITOR,
            blue_restore_host('Enterprise0'),
        ]

        red_policy = scripted_red_policy_factory(red_actions)
        blue_policy = scripted_blue_policy_factory(blue_actions)

        result = harness.run_episode(blue_policy, red_policy)

        restore_step = result.step_results[1]

        assert abs(restore_step.cyborg_state.reward_blue - restore_step.jax_state.reward_blue) < 0.02, \
            f"Blue reward mismatch on clean host Restore: CybORG={restore_step.cyborg_state.reward_blue}, JAX={restore_step.jax_state.reward_blue}"

    def test_restore_vs_red_reinfection(self):
        """CybORG vs JAX: Red can re-exploit host after Restore."""
        harness = DifferentialHarness(seed=42, max_steps=12, verbose=False)

        red_actions = [
            red_discover_subnet(SUBNET_USER),
            red_scan_host('User3'),
            red_exploit_host('User3', EXPLOIT_SSH),
            0,  # Sleep
            red_scan_host('User3'),
            red_exploit_host('User3', EXPLOIT_SSH),
            0,
        ]

        blue_actions = [
            BLUE_MONITOR,
            BLUE_MONITOR,
            BLUE_MONITOR,
            blue_restore_host('User3'),
            BLUE_MONITOR,
            BLUE_MONITOR,
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

    @pytest.mark.parametrize("host", ['User1', 'User2', 'User3', 'User4'])
    def test_restore_on_different_user_hosts(self, host):
        """CybORG vs JAX: Restore behavior should match across different hosts."""
        harness = DifferentialHarness(seed=42, max_steps=8, verbose=False)

        red_actions = [
            red_discover_subnet(SUBNET_USER),
            red_scan_host(host),
            red_exploit_host(host, EXPLOIT_SSH),
            red_privesc_host(host),
            0,
        ]

        blue_actions = [
            BLUE_MONITOR,
            BLUE_MONITOR,
            BLUE_MONITOR,
            BLUE_MONITOR,
            blue_restore_host(host),
        ]

        red_policy = scripted_red_policy_factory(red_actions)
        blue_policy = scripted_blue_policy_factory(blue_actions)

        result = harness.run_episode(blue_policy, red_policy)

        restore_step = result.step_results[4]
        assert abs(restore_step.cyborg_state.reward_blue - restore_step.jax_state.reward_blue) < 0.02, \
            f"Restore on {host}: Blue reward mismatch CybORG={restore_step.cyborg_state.reward_blue}, JAX={restore_step.jax_state.reward_blue}"

    def test_multiple_restores_in_episode(self):
        """CybORG vs JAX: Multiple Restore actions in one episode."""
        harness = DifferentialHarness(seed=42, max_steps=15, verbose=False)

        red_actions = [
            red_discover_subnet(SUBNET_USER),
            red_scan_host('User1'),
            red_exploit_host('User1', EXPLOIT_SSH),
            red_scan_host('User2'),
            red_exploit_host('User2', EXPLOIT_SSH),
            0,
            0,
            0,
        ]

        blue_actions = [
            BLUE_MONITOR,
            BLUE_MONITOR,
            BLUE_MONITOR,
            BLUE_MONITOR,
            BLUE_MONITOR,
            blue_restore_host('User1'),
            blue_restore_host('User2'),
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

    def test_restore_after_analyse(self):
        """CybORG vs JAX: Restore after Analyse has detected activity."""
        harness = DifferentialHarness(seed=42, max_steps=10, verbose=False)

        red_actions = [
            red_discover_subnet(SUBNET_USER),
            red_scan_host('User2'),
            red_exploit_host('User2', EXPLOIT_SSH),
            red_privesc_host('User2'),
            0,
            0,
        ]

        blue_actions = [
            BLUE_MONITOR,
            BLUE_MONITOR,
            BLUE_MONITOR,
            BLUE_MONITOR,
            blue_analyse_host('User2'),
            blue_restore_host('User2'),
        ]

        red_policy = scripted_red_policy_factory(red_actions)
        blue_policy = scripted_blue_policy_factory(blue_actions)

        result = harness.run_episode(blue_policy, red_policy)

        for sr in result.step_results:
            cyborg_blue = sr.cyborg_state.reward_blue
            jax_blue = sr.jax_state.reward_blue
            assert abs(cyborg_blue - jax_blue) < 0.02, \
                f"Step {sr.step}: Blue reward mismatch CybORG={cyborg_blue}, JAX={jax_blue}"

    def test_restore_clears_sessions_completely(self):
        """CybORG vs JAX: Restore removes all Red sessions on host."""
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
            blue_restore_host('User1'),
        ]

        red_policy = scripted_red_policy_factory(red_actions)
        blue_policy = scripted_blue_policy_factory(blue_actions)

        result = harness.run_episode(blue_policy, red_policy)

        before_restore = result.step_results[3]
        after_restore = result.step_results[4]

        assert before_restore.jax_state.red_privilege.get('User1', 0) >= COMPROMISE_USER, \
            "Red should have privilege before Restore"

        assert after_restore.jax_state.red_privilege.get('User1', 0) == 0, \
            "Red should have no privilege after Restore"

        assert after_restore.jax_state.red_sessions.get('User1', 0) == 0, \
            "Red should have no sessions after Restore"
