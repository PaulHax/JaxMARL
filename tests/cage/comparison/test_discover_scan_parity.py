"""Differential tests comparing CybORG Discovery and Scan behavior to JAX.

Every test runs both CybORG and JAX with identical actions and compares results.
These tests verify DiscoverRemoteSystems and DiscoverNetworkServices behavior.
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
    SUBNET_USER,
    SUBNET_ENTERPRISE,
    SUBNET_OPERATIONAL,
    EXPLOIT_SSH,
)
from jaxmarl.environments.cage.actions import BLUE_MONITOR


requires_cyborg = pytest.mark.skipif(
    not is_cyborg_available(),
    reason="CybORG not installed"
)


@requires_cyborg
class TestDiscoverSubnetCybORGParity:
    """Differential tests for DiscoverRemoteSystems (subnet discovery)."""

    def test_discover_user_subnet(self):
        """CybORG vs JAX: Discover User subnet."""
        harness = DifferentialHarness(seed=42, max_steps=5, verbose=False)

        red_actions = [
            red_discover_subnet(SUBNET_USER),
            0,
        ]

        blue_actions = [BLUE_MONITOR] * 2

        red_policy = scripted_red_policy_factory(red_actions)
        blue_policy = scripted_blue_policy_factory(blue_actions)

        result = harness.run_episode(blue_policy, red_policy)

        for sr in result.step_results:
            cyborg_blue = sr.cyborg_state.reward_blue
            jax_blue = sr.jax_state.reward_blue
            assert abs(cyborg_blue - jax_blue) < 0.02, \
                f"Step {sr.step}: Blue reward mismatch CybORG={cyborg_blue}, JAX={jax_blue}"

    def test_discover_enterprise_subnet_without_prerequisite(self):
        """CybORG vs JAX: Discover Enterprise subnet without User access."""
        harness = DifferentialHarness(seed=42, max_steps=5, verbose=False)

        red_actions = [
            red_discover_subnet(SUBNET_ENTERPRISE),
            0,
        ]

        blue_actions = [BLUE_MONITOR] * 2

        red_policy = scripted_red_policy_factory(red_actions)
        blue_policy = scripted_blue_policy_factory(blue_actions)

        result = harness.run_episode(blue_policy, red_policy)

        for sr in result.step_results:
            cyborg_blue = sr.cyborg_state.reward_blue
            jax_blue = sr.jax_state.reward_blue
            assert abs(cyborg_blue - jax_blue) < 0.02, \
                f"Step {sr.step}: Blue reward mismatch CybORG={cyborg_blue}, JAX={jax_blue}"

    def test_discover_enterprise_subnet_with_prerequisite(self):
        """CybORG vs JAX: Discover Enterprise subnet after User compromise."""
        harness = DifferentialHarness(seed=42, max_steps=10, verbose=False)

        red_actions = [
            red_discover_subnet(SUBNET_USER),
            red_scan_host('User1'),
            red_exploit_host('User1', EXPLOIT_SSH),
            red_privesc_host('User1'),
            red_discover_subnet(SUBNET_ENTERPRISE),
            0,
        ]

        blue_actions = [BLUE_MONITOR] * 6

        red_policy = scripted_red_policy_factory(red_actions)
        blue_policy = scripted_blue_policy_factory(blue_actions)

        result = harness.run_episode(blue_policy, red_policy)

        for sr in result.step_results:
            cyborg_blue = sr.cyborg_state.reward_blue
            jax_blue = sr.jax_state.reward_blue
            assert abs(cyborg_blue - jax_blue) < 0.02, \
                f"Step {sr.step}: Blue reward mismatch CybORG={cyborg_blue}, JAX={jax_blue}"

    def test_discover_operational_subnet(self):
        """CybORG vs JAX: Discover Operational subnet (requires Enterprise access)."""
        harness = DifferentialHarness(seed=42, max_steps=15, verbose=False)

        red_actions = [
            red_discover_subnet(SUBNET_USER),
            red_scan_host('User1'),
            red_exploit_host('User1', EXPLOIT_SSH),
            red_privesc_host('User1'),
            red_discover_subnet(SUBNET_ENTERPRISE),
            red_scan_host('Enterprise1'),
            red_exploit_host('Enterprise1', EXPLOIT_SSH),
            red_privesc_host('Enterprise1'),
            red_discover_subnet(SUBNET_OPERATIONAL),
            0,
        ]

        blue_actions = [BLUE_MONITOR] * 10

        red_policy = scripted_red_policy_factory(red_actions)
        blue_policy = scripted_blue_policy_factory(blue_actions)

        result = harness.run_episode(blue_policy, red_policy)

        for sr in result.step_results:
            cyborg_blue = sr.cyborg_state.reward_blue
            jax_blue = sr.jax_state.reward_blue
            assert abs(cyborg_blue - jax_blue) < 0.02, \
                f"Step {sr.step}: Blue reward mismatch CybORG={cyborg_blue}, JAX={jax_blue}"

    def test_discover_same_subnet_twice(self):
        """CybORG vs JAX: Discover same subnet twice (second is no-op)."""
        harness = DifferentialHarness(seed=42, max_steps=5, verbose=False)

        red_actions = [
            red_discover_subnet(SUBNET_USER),
            red_discover_subnet(SUBNET_USER),
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


@requires_cyborg
class TestScanHostCybORGParity:
    """Differential tests for DiscoverNetworkServices (host scanning)."""

    def test_scan_host_after_discover(self):
        """CybORG vs JAX: Scan host after discovering subnet."""
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

    def test_scan_host_without_discover(self):
        """CybORG vs JAX: Scan host without discovering subnet first."""
        harness = DifferentialHarness(seed=42, max_steps=5, verbose=False)

        red_actions = [
            red_scan_host('User1'),
            0,
        ]

        blue_actions = [BLUE_MONITOR] * 2

        red_policy = scripted_red_policy_factory(red_actions)
        blue_policy = scripted_blue_policy_factory(blue_actions)

        result = harness.run_episode(blue_policy, red_policy)

        for sr in result.step_results:
            cyborg_blue = sr.cyborg_state.reward_blue
            jax_blue = sr.jax_state.reward_blue
            assert abs(cyborg_blue - jax_blue) < 0.02, \
                f"Step {sr.step}: Blue reward mismatch CybORG={cyborg_blue}, JAX={jax_blue}"

    @pytest.mark.parametrize("host", ['User1', 'User2', 'User3', 'User4'])
    def test_scan_different_user_hosts(self, host):
        """CybORG vs JAX: Scan different User hosts."""
        harness = DifferentialHarness(seed=42, max_steps=5, verbose=False)

        red_actions = [
            red_discover_subnet(SUBNET_USER),
            red_scan_host(host),
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
                f"Scan {host} step {sr.step}: Blue reward mismatch CybORG={cyborg_blue}, JAX={jax_blue}"

    def test_scan_multiple_hosts(self):
        """CybORG vs JAX: Scan multiple hosts in sequence."""
        harness = DifferentialHarness(seed=42, max_steps=10, verbose=False)

        red_actions = [
            red_discover_subnet(SUBNET_USER),
            red_scan_host('User1'),
            red_scan_host('User2'),
            red_scan_host('User3'),
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

    def test_scan_same_host_twice(self):
        """CybORG vs JAX: Scan same host twice."""
        harness = DifferentialHarness(seed=42, max_steps=5, verbose=False)

        red_actions = [
            red_discover_subnet(SUBNET_USER),
            red_scan_host('User1'),
            red_scan_host('User1'),
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

    def test_scan_enterprise_host(self):
        """CybORG vs JAX: Scan Enterprise host after lateral movement setup."""
        harness = DifferentialHarness(seed=42, max_steps=10, verbose=False)

        red_actions = [
            red_discover_subnet(SUBNET_USER),
            red_scan_host('User1'),
            red_exploit_host('User1', EXPLOIT_SSH),
            red_privesc_host('User1'),
            red_scan_host('Enterprise0'),
            0,
        ]

        blue_actions = [BLUE_MONITOR] * 6

        red_policy = scripted_red_policy_factory(red_actions)
        blue_policy = scripted_blue_policy_factory(blue_actions)

        result = harness.run_episode(blue_policy, red_policy)

        for sr in result.step_results:
            cyborg_blue = sr.cyborg_state.reward_blue
            jax_blue = sr.jax_state.reward_blue
            assert abs(cyborg_blue - jax_blue) < 0.02, \
                f"Step {sr.step}: Blue reward mismatch CybORG={cyborg_blue}, JAX={jax_blue}"


@requires_cyborg
class TestDiscoverScanSequenceParity:
    """Tests for complete discover->scan->exploit sequences."""

    def test_full_discovery_sequence_user(self):
        """CybORG vs JAX: Full discover->scan->exploit on User host."""
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

    def test_out_of_order_actions(self):
        """CybORG vs JAX: Out of order actions (scan before discover)."""
        harness = DifferentialHarness(seed=42, max_steps=8, verbose=False)

        red_actions = [
            red_scan_host('User1'),
            red_discover_subnet(SUBNET_USER),
            red_scan_host('User1'),
            red_exploit_host('User1', EXPLOIT_SSH),
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

    def test_exploit_before_scan_fails(self):
        """CybORG vs JAX: Exploit before scan should fail."""
        harness = DifferentialHarness(seed=42, max_steps=5, verbose=False)

        red_actions = [
            red_discover_subnet(SUBNET_USER),
            red_exploit_host('User1', EXPLOIT_SSH),
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
