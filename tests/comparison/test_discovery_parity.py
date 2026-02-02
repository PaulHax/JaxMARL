"""Differential tests for discovery/scan semantics.

Tests exploring whether CybORG allows exploit with only DiscoverSubnet
vs requiring explicit Scan.
"""

import pytest
from tests.differential.harness import DifferentialHarness, is_cyborg_available
from tests.comparison.policies import scripted_red_policy_factory, sleep_policy
from tests.comparison.scenarios import (
    red_discover_subnet, red_scan_host, red_exploit_host, red_privesc_host,
    SUBNET_USER, EXPLOIT_SSH,
)
from jaxmarl.environments.cage.actions import RED_SLEEP


requires_cyborg = pytest.mark.skipif(
    not is_cyborg_available(),
    reason="CybORG not installed"
)


@requires_cyborg
class TestDiscoveryScanSemantics:
    """Test discovery and scan requirements for exploit."""

    @pytest.fixture
    def harness(self):
        return DifferentialHarness(seed=42, max_steps=10, verbose=True)

    def test_exploit_with_scan_succeeds(self, harness):
        """Baseline: Exploit with proper scan should succeed in both."""
        red_actions = [
            red_discover_subnet(SUBNET_USER),
            red_scan_host('User1'),
            red_exploit_host('User1', EXPLOIT_SSH),
        ]

        red_policy = scripted_red_policy_factory(red_actions)
        result = harness.run_episode(sleep_policy, red_policy)

        exploit_step = result.step_results[2]

        print(f"\n=== Exploit WITH Scan ===")
        print(f"CybORG reward: {exploit_step.cyborg_state.reward_red}")
        print(f"JAX reward: {exploit_step.jax_state.reward_red}")
        print(f"CybORG User1 compromised: {exploit_step.cyborg_state.host_compromised.get('User1', False)}")
        print(f"JAX User1 compromised: {exploit_step.jax_state.host_compromised.get('User1', False)}")

        assert exploit_step.cyborg_state.host_compromised.get('User1', False), \
            "CybORG should compromise User1 with scan"
        assert exploit_step.jax_state.host_compromised.get('User1', False), \
            "JAX should compromise User1 with scan"

    def test_exploit_without_scan_cyborg_behavior(self, harness):
        """Test CybORG's actual behavior when exploiting without scan.

        This test documents CybORG's behavior - we need to verify if:
        1. CybORG allows exploit after DiscoverSubnet only
        2. Or if CybORG also requires scan
        """
        red_actions = [
            red_discover_subnet(SUBNET_USER),
            red_exploit_host('User1', EXPLOIT_SSH),  # Skip scan
        ]

        red_policy = scripted_red_policy_factory(red_actions)
        result = harness.run_episode(sleep_policy, red_policy)

        exploit_step = result.step_results[1]

        print(f"\n=== Exploit WITHOUT Scan ===")
        print(f"CybORG reward: {exploit_step.cyborg_state.reward_red}")
        print(f"JAX reward: {exploit_step.jax_state.reward_red}")
        print(f"CybORG User1 compromised: {exploit_step.cyborg_state.host_compromised.get('User1', False)}")
        print(f"JAX User1 compromised: {exploit_step.jax_state.host_compromised.get('User1', False)}")
        print(f"CybORG User1 privilege: {exploit_step.cyborg_state.red_privilege.get('User1', 0)}")
        print(f"JAX User1 privilege: {exploit_step.jax_state.red_privilege.get('User1', 0)}")

        cyborg_succeeded = exploit_step.cyborg_state.host_compromised.get('User1', False)
        jax_succeeded = exploit_step.jax_state.host_compromised.get('User1', False)

        if cyborg_succeeded and not jax_succeeded:
            print("\n!!! CybORG allows exploit without scan, JAX does not !!!")
        elif not cyborg_succeeded and not jax_succeeded:
            print("\n=== Both require scan - no semantic difference ===")
        elif cyborg_succeeded and jax_succeeded:
            print("\n=== Both allow exploit without scan ===")

        assert cyborg_succeeded == jax_succeeded, \
            f"Exploit-without-scan behavior differs: CybORG={cyborg_succeeded}, JAX={jax_succeeded}"

    def test_exploit_without_discover_fails(self, harness):
        """Exploit without any discovery should fail in both."""
        red_actions = [
            red_exploit_host('User1', EXPLOIT_SSH),  # No discover, no scan
        ]

        red_policy = scripted_red_policy_factory(red_actions)
        result = harness.run_episode(sleep_policy, red_policy)

        exploit_step = result.step_results[0]

        print(f"\n=== Exploit WITHOUT Discover ===")
        print(f"CybORG reward: {exploit_step.cyborg_state.reward_red}")
        print(f"JAX reward: {exploit_step.jax_state.reward_red}")
        print(f"CybORG User1 compromised: {exploit_step.cyborg_state.host_compromised.get('User1', False)}")
        print(f"JAX User1 compromised: {exploit_step.jax_state.host_compromised.get('User1', False)}")

        assert not exploit_step.cyborg_state.host_compromised.get('User1', False), \
            "CybORG should NOT compromise User1 without discover"
        assert not exploit_step.jax_state.host_compromised.get('User1', False), \
            "JAX should NOT compromise User1 without discover"

    def test_cross_subnet_scan_enables_exploit(self, harness):
        """Test that scan from adjacent subnet enables exploit without DiscoverSubnet.

        CybORG allows scan if Red has a session in an adjacent subnet. The scan
        reveals host info needed for exploit. B_line uses this pattern to scan
        Enterprise1 from User1 session without calling DiscoverSubnet(Enterprise).
        """
        red_actions = [
            red_discover_subnet(SUBNET_USER),
            red_scan_host('User1'),
            red_exploit_host('User1', EXPLOIT_SSH),
            red_privesc_host('User1'),
            red_scan_host('Enterprise1'),  # Cross-subnet scan without DiscoverSubnet(Enterprise)
            red_exploit_host('Enterprise1', EXPLOIT_SSH),
        ]

        red_policy = scripted_red_policy_factory(red_actions)
        result = harness.run_episode(sleep_policy, red_policy)

        exploit_step = result.step_results[5]

        cyborg_compromised = exploit_step.cyborg_state.host_compromised.get('Enterprise1', 0)
        jax_compromised = exploit_step.jax_state.host_compromised.get('Enterprise1', 0)

        assert cyborg_compromised > 0, "CybORG should compromise Enterprise1 after cross-subnet scan+exploit"
        assert jax_compromised > 0, "JAX should compromise Enterprise1 after cross-subnet scan+exploit"
        assert cyborg_compromised == jax_compromised, \
            f"Compromise level differs: CybORG={cyborg_compromised}, JAX={jax_compromised}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
