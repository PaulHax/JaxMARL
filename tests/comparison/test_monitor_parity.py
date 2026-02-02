"""Differential tests comparing CybORG Monitor behavior to JAX.

Every test runs both CybORG and JAX with identical actions and compares results.
These tests verify Monitor detection behavior and observation updates.
"""

import pytest
from tests.differential.harness import DifferentialHarness, is_cyborg_available
from tests.comparison.policies import (
    scripted_blue_policy_factory,
    scripted_red_policy_factory,
)
from tests.comparison.scenarios import (
    red_discover_subnet,
    red_scan_host,
    red_exploit_host,
    red_privesc_host,
    blue_remove_host,
    SUBNET_USER,
    EXPLOIT_SSH,
    BLINE_KILLCHAIN_STANDARD,
)
from jaxmarl.environments.cage.actions import BLUE_SLEEP, BLUE_MONITOR


requires_cyborg = pytest.mark.skipif(
    not is_cyborg_available(),
    reason="CybORG not installed"
)


@requires_cyborg
class TestMonitorCybORGParity:
    """Differential tests comparing CybORG Monitor behavior to JAX.

    Key CybORG behavior:
    - Monitor is a passive action with no direct effect
    - Monitor updates Blue's observations based on network activity
    - Monitor has no action cost
    - Red's scanning/exploiting activity may be detected via Monitor
    """

    def test_monitor_baseline(self):
        """CybORG vs JAX: Monitor with no Red activity."""
        harness = DifferentialHarness(seed=42, max_steps=5, verbose=False)

        red_actions = [0, 0, 0, 0, 0]

        blue_actions = [BLUE_MONITOR] * 5

        red_policy = scripted_red_policy_factory(red_actions)
        blue_policy = scripted_blue_policy_factory(blue_actions)

        result = harness.run_episode(blue_policy, red_policy)

        for sr in result.step_results:
            cyborg_blue = sr.cyborg_state.reward_blue
            jax_blue = sr.jax_state.reward_blue
            assert abs(cyborg_blue - jax_blue) < 0.02, \
                f"Step {sr.step}: Blue reward mismatch CybORG={cyborg_blue}, JAX={jax_blue}"

    def test_monitor_during_red_discovery(self):
        """CybORG vs JAX: Monitor while Red discovers subnet."""
        harness = DifferentialHarness(seed=42, max_steps=5, verbose=False)

        red_actions = [
            red_discover_subnet(SUBNET_USER),
            0,
            0,
        ]

        blue_actions = [BLUE_MONITOR] * 3

        red_policy = scripted_red_policy_factory(red_actions)
        blue_policy = scripted_blue_policy_factory(blue_actions)

        result = harness.run_episode(blue_policy, red_policy)

        for sr in result.step_results:
            cyborg_blue = sr.cyborg_state.reward_blue
            jax_blue = sr.jax_state.reward_blue
            assert abs(cyborg_blue - jax_blue) < 0.02, \
                f"Step {sr.step}: Blue reward mismatch CybORG={cyborg_blue}, JAX={jax_blue}"

    def test_monitor_during_red_scan(self):
        """CybORG vs JAX: Monitor while Red scans host."""
        harness = DifferentialHarness(seed=42, max_steps=5, verbose=False)

        red_actions = [
            red_discover_subnet(SUBNET_USER),
            red_scan_host('User1'),
            0,
        ]

        blue_actions = [BLUE_MONITOR] * 3

        red_policy = scripted_red_policy_factory(red_actions)
        blue_policy = scripted_blue_policy_factory(blue_actions)

        result = harness.run_episode(blue_policy, red_policy)

        for sr in result.step_results:
            cyborg_blue = sr.cyborg_state.reward_blue
            jax_blue = sr.jax_state.reward_blue
            assert abs(cyborg_blue - jax_blue) < 0.02, \
                f"Step {sr.step}: Blue reward mismatch CybORG={cyborg_blue}, JAX={jax_blue}"

    def test_monitor_during_red_exploit(self):
        """CybORG vs JAX: Monitor while Red exploits host."""
        harness = DifferentialHarness(seed=42, max_steps=8, verbose=False)

        red_actions = [
            red_discover_subnet(SUBNET_USER),
            red_scan_host('User1'),
            red_exploit_host('User1', EXPLOIT_SSH),
            0,
        ]

        blue_actions = [BLUE_MONITOR] * 4

        red_policy = scripted_red_policy_factory(red_actions)
        blue_policy = scripted_blue_policy_factory(blue_actions)

        result = harness.run_episode(blue_policy, red_policy)

        for sr in result.step_results:
            cyborg_blue = sr.cyborg_state.reward_blue
            jax_blue = sr.jax_state.reward_blue
            assert abs(cyborg_blue - jax_blue) < 0.02, \
                f"Step {sr.step}: Blue reward mismatch CybORG={cyborg_blue}, JAX={jax_blue}"

    def test_monitor_during_red_privesc(self):
        """CybORG vs JAX: Monitor while Red escalates privilege."""
        harness = DifferentialHarness(seed=42, max_steps=10, verbose=False)

        red_actions = [
            red_discover_subnet(SUBNET_USER),
            red_scan_host('User1'),
            red_exploit_host('User1', EXPLOIT_SSH),
            red_privesc_host('User1'),
            0,
        ]

        blue_actions = [BLUE_MONITOR] * 5

        red_policy = scripted_red_policy_factory(red_actions)
        blue_policy = scripted_blue_policy_factory(blue_actions)

        result = harness.run_episode(blue_policy, red_policy)

        for sr in result.step_results:
            cyborg_blue = sr.cyborg_state.reward_blue
            jax_blue = sr.jax_state.reward_blue
            assert abs(cyborg_blue - jax_blue) < 0.02, \
                f"Step {sr.step}: Blue reward mismatch CybORG={cyborg_blue}, JAX={jax_blue}"

    def test_monitor_throughout_killchain(self):
        """CybORG vs JAX: Monitor through entire B_line killchain."""
        harness = DifferentialHarness(seed=42, max_steps=20, verbose=False)

        red_actions = BLINE_KILLCHAIN_STANDARD[:16]

        blue_actions = [BLUE_MONITOR] * 16

        red_policy = scripted_red_policy_factory(red_actions)
        blue_policy = scripted_blue_policy_factory(blue_actions)

        result = harness.run_episode(blue_policy, red_policy)

        for sr in result.step_results:
            cyborg_blue = sr.cyborg_state.reward_blue
            jax_blue = sr.jax_state.reward_blue
            assert abs(cyborg_blue - jax_blue) < 0.02, \
                f"Step {sr.step}: Blue reward mismatch CybORG={cyborg_blue}, JAX={jax_blue}"

    def test_monitor_vs_sleep(self):
        """CybORG vs JAX: Compare Monitor and Sleep (both passive)."""
        harness = DifferentialHarness(seed=42, max_steps=10, verbose=False)

        red_actions = [
            red_discover_subnet(SUBNET_USER),
            red_scan_host('User1'),
            red_exploit_host('User1', EXPLOIT_SSH),
            0,
            0,
        ]

        blue_monitor = [BLUE_MONITOR] * 5
        blue_sleep = [BLUE_SLEEP] * 5

        red_policy = scripted_red_policy_factory(red_actions)
        blue_monitor_policy = scripted_blue_policy_factory(blue_monitor)
        blue_sleep_policy = scripted_blue_policy_factory(blue_sleep)

        result_monitor = harness.run_episode(blue_monitor_policy, red_policy)

        harness2 = DifferentialHarness(seed=42, max_steps=10, verbose=False)
        result_sleep = harness2.run_episode(blue_sleep_policy, scripted_red_policy_factory(red_actions))

        for sr in result_monitor.step_results:
            cyborg_blue = sr.cyborg_state.reward_blue
            jax_blue = sr.jax_state.reward_blue
            assert abs(cyborg_blue - jax_blue) < 0.02, \
                f"Monitor step {sr.step}: Blue reward mismatch CybORG={cyborg_blue}, JAX={jax_blue}"

        for sr in result_sleep.step_results:
            cyborg_blue = sr.cyborg_state.reward_blue
            jax_blue = sr.jax_state.reward_blue
            assert abs(cyborg_blue - jax_blue) < 0.02, \
                f"Sleep step {sr.step}: Blue reward mismatch CybORG={cyborg_blue}, JAX={jax_blue}"

    def test_monitor_multiple_hosts_compromised(self):
        """CybORG vs JAX: Monitor while Red compromises multiple hosts."""
        harness = DifferentialHarness(seed=42, max_steps=15, verbose=False)

        red_actions = [
            red_discover_subnet(SUBNET_USER),
            red_scan_host('User1'),
            red_exploit_host('User1', EXPLOIT_SSH),
            red_scan_host('User2'),
            red_exploit_host('User2', EXPLOIT_SSH),
            red_scan_host('User3'),
            red_exploit_host('User3', EXPLOIT_SSH),
            0,
        ]

        blue_actions = [BLUE_MONITOR] * 8

        red_policy = scripted_red_policy_factory(red_actions)
        blue_policy = scripted_blue_policy_factory(blue_actions)

        result = harness.run_episode(blue_policy, red_policy)

        for sr in result.step_results:
            cyborg_blue = sr.cyborg_state.reward_blue
            jax_blue = sr.jax_state.reward_blue
            assert abs(cyborg_blue - jax_blue) < 0.02, \
                f"Step {sr.step}: Blue reward mismatch CybORG={cyborg_blue}, JAX={jax_blue}"

    def test_monitor_after_remove(self):
        """CybORG vs JAX: Monitor after Remove action."""
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
            blue_remove_host('User1'),
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

    def test_monitor_reward_is_zero(self):
        """CybORG vs JAX: Monitor should have no action cost."""
        harness = DifferentialHarness(seed=42, max_steps=5, verbose=False)

        red_actions = [0, 0, 0]

        blue_actions = [BLUE_MONITOR, BLUE_MONITOR, BLUE_MONITOR]

        red_policy = scripted_red_policy_factory(red_actions)
        blue_policy = scripted_blue_policy_factory(blue_actions)

        result = harness.run_episode(blue_policy, red_policy)

        for sr in result.step_results:
            assert sr.jax_state.reward_blue == 0.0, \
                f"Step {sr.step}: Monitor should have zero reward, got {sr.jax_state.reward_blue}"
