"""Differential tests comparing CybORG Analyse behavior to JAX.

Every test runs both CybORG and JAX with identical actions and compares results.
These tests verify Analyse detection behavior and observation updates.
"""

import pytest
from tests.cage.differential.harness import DifferentialHarness, is_cyborg_available
from tests.cage.comparison.policies import (
    scripted_blue_policy_factory,
    scripted_red_policy_factory,
)
from tests.cage.comparison.scenarios import (
    red_discover_subnet,
    red_scan_host,
    red_exploit_host,
    red_privesc_host,
    blue_analyse_host,
    blue_remove_host,
    blue_restore_host,
    SUBNET_USER,
    EXPLOIT_SSH,
)
from jaxmarl.environments.cage.actions import BLUE_MONITOR


requires_cyborg = pytest.mark.skipif(
    not is_cyborg_available(),
    reason="CybORG not installed"
)


@requires_cyborg
class TestAnalyseCybORGParity:
    """Differential tests comparing CybORG Analyse behavior to JAX.

    Key CybORG behavior:
    - Analyse detects if there's Red activity on the host
    - Analyse updates Blue's observation with compromise information
    - Analyse has no direct action cost
    - Analyse reveals privilege level (user vs privileged)
    """

    def test_analyse_on_compromised_host(self):
        """CybORG vs JAX: Analyse on USER-compromised host detects activity."""
        harness = DifferentialHarness(seed=42, max_steps=10, verbose=False)

        red_actions = [
            red_discover_subnet(SUBNET_USER),
            red_scan_host('User1'),
            red_exploit_host('User1', EXPLOIT_SSH),
            0,
            0,
        ]

        blue_actions = [
            BLUE_MONITOR,
            BLUE_MONITOR,
            BLUE_MONITOR,
            blue_analyse_host('User1'),
            BLUE_MONITOR,
        ]

        red_policy = scripted_red_policy_factory(red_actions)
        blue_policy = scripted_blue_policy_factory(blue_actions)

        result = harness.run_episode(blue_policy, red_policy)

        analyse_step = result.step_results[3]

        assert abs(analyse_step.cyborg_state.reward_blue - analyse_step.jax_state.reward_blue) < 0.02, \
            f"Blue reward mismatch at Analyse: CybORG={analyse_step.cyborg_state.reward_blue}, JAX={analyse_step.jax_state.reward_blue}"

    def test_analyse_on_privileged_host(self):
        """CybORG vs JAX: Analyse on PRIVILEGED host detects elevated access."""
        harness = DifferentialHarness(seed=42, max_steps=10, verbose=False)

        red_actions = [
            red_discover_subnet(SUBNET_USER),
            red_scan_host('User1'),
            red_exploit_host('User1', EXPLOIT_SSH),
            red_privesc_host('User1'),
            0,
            0,
        ]

        blue_actions = [
            BLUE_MONITOR,
            BLUE_MONITOR,
            BLUE_MONITOR,
            BLUE_MONITOR,
            blue_analyse_host('User1'),
            BLUE_MONITOR,
        ]

        red_policy = scripted_red_policy_factory(red_actions)
        blue_policy = scripted_blue_policy_factory(blue_actions)

        result = harness.run_episode(blue_policy, red_policy)

        analyse_step = result.step_results[4]

        assert abs(analyse_step.cyborg_state.reward_blue - analyse_step.jax_state.reward_blue) < 0.02, \
            f"Blue reward mismatch at Analyse: CybORG={analyse_step.cyborg_state.reward_blue}, JAX={analyse_step.jax_state.reward_blue}"

    def test_analyse_on_clean_host(self):
        """CybORG vs JAX: Analyse on non-compromised host finds nothing."""
        harness = DifferentialHarness(seed=42, max_steps=5, verbose=False)

        red_actions = [
            0,
            0,
        ]

        blue_actions = [
            blue_analyse_host('Enterprise0'),
            BLUE_MONITOR,
        ]

        red_policy = scripted_red_policy_factory(red_actions)
        blue_policy = scripted_blue_policy_factory(blue_actions)

        result = harness.run_episode(blue_policy, red_policy)

        analyse_step = result.step_results[0]

        assert abs(analyse_step.cyborg_state.reward_blue - analyse_step.jax_state.reward_blue) < 0.02, \
            f"Blue reward mismatch on clean host: CybORG={analyse_step.cyborg_state.reward_blue}, JAX={analyse_step.jax_state.reward_blue}"

    @pytest.mark.parametrize("host", ['User1', 'User2', 'User3', 'User4'])
    def test_analyse_on_different_user_hosts(self, host):
        """CybORG vs JAX: Analyse behavior should match across User hosts."""
        harness = DifferentialHarness(seed=42, max_steps=8, verbose=False)

        red_actions = [
            red_discover_subnet(SUBNET_USER),
            red_scan_host(host),
            red_exploit_host(host, EXPLOIT_SSH),
            0,
        ]

        blue_actions = [
            BLUE_MONITOR,
            BLUE_MONITOR,
            BLUE_MONITOR,
            blue_analyse_host(host),
        ]

        red_policy = scripted_red_policy_factory(red_actions)
        blue_policy = scripted_blue_policy_factory(blue_actions)

        result = harness.run_episode(blue_policy, red_policy)

        analyse_step = result.step_results[3]
        assert abs(analyse_step.cyborg_state.reward_blue - analyse_step.jax_state.reward_blue) < 0.02, \
            f"Analyse on {host}: Blue reward mismatch CybORG={analyse_step.cyborg_state.reward_blue}, JAX={analyse_step.jax_state.reward_blue}"

    def test_analyse_then_remove(self):
        """CybORG vs JAX: Analyse followed by Remove."""
        harness = DifferentialHarness(seed=42, max_steps=10, verbose=False)

        red_actions = [
            red_discover_subnet(SUBNET_USER),
            red_scan_host('User2'),
            red_exploit_host('User2', EXPLOIT_SSH),
            0,
            0,
        ]

        blue_actions = [
            BLUE_MONITOR,
            BLUE_MONITOR,
            BLUE_MONITOR,
            blue_analyse_host('User2'),
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

    def test_analyse_then_restore(self):
        """CybORG vs JAX: Analyse followed by Restore."""
        harness = DifferentialHarness(seed=42, max_steps=10, verbose=False)

        red_actions = [
            red_discover_subnet(SUBNET_USER),
            red_scan_host('User3'),
            red_exploit_host('User3', EXPLOIT_SSH),
            red_privesc_host('User3'),
            0,
            0,
        ]

        blue_actions = [
            BLUE_MONITOR,
            BLUE_MONITOR,
            BLUE_MONITOR,
            BLUE_MONITOR,
            blue_analyse_host('User3'),
            blue_restore_host('User3'),
        ]

        red_policy = scripted_red_policy_factory(red_actions)
        blue_policy = scripted_blue_policy_factory(blue_actions)

        result = harness.run_episode(blue_policy, red_policy)

        for sr in result.step_results:
            cyborg_blue = sr.cyborg_state.reward_blue
            jax_blue = sr.jax_state.reward_blue
            assert abs(cyborg_blue - jax_blue) < 0.02, \
                f"Step {sr.step}: Blue reward mismatch CybORG={cyborg_blue}, JAX={jax_blue}"

    def test_multiple_analyses_same_host(self):
        """CybORG vs JAX: Multiple Analyse actions on same host."""
        harness = DifferentialHarness(seed=42, max_steps=10, verbose=False)

        red_actions = [
            red_discover_subnet(SUBNET_USER),
            red_scan_host('User1'),
            red_exploit_host('User1', EXPLOIT_SSH),
            0,
            0,
            0,
        ]

        blue_actions = [
            BLUE_MONITOR,
            BLUE_MONITOR,
            BLUE_MONITOR,
            blue_analyse_host('User1'),
            blue_analyse_host('User1'),
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

    def test_analyse_different_hosts_sequentially(self):
        """CybORG vs JAX: Analyse multiple hosts in sequence."""
        harness = DifferentialHarness(seed=42, max_steps=12, verbose=False)

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
            blue_analyse_host('User1'),
            blue_analyse_host('User2'),
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

    def test_analyse_after_red_leaves(self):
        """CybORG vs JAX: Analyse after Red has been removed."""
        harness = DifferentialHarness(seed=42, max_steps=12, verbose=False)

        red_actions = [
            red_discover_subnet(SUBNET_USER),
            red_scan_host('User1'),
            red_exploit_host('User1', EXPLOIT_SSH),
            0,
            0,
            0,
        ]

        blue_actions = [
            BLUE_MONITOR,
            BLUE_MONITOR,
            BLUE_MONITOR,
            blue_remove_host('User1'),
            blue_analyse_host('User1'),
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
