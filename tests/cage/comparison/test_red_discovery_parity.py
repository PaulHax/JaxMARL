"""Tests verifying red_discovered_hosts parity between CybORG and JAX.

CybORG considers a host "discovered" once Red has observed it via scan
(DiscoverNetworkServices) or successful exploit, even without a prior
DiscoverRemoteSystems.  JAX must match this behavior.
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
    EXPLOIT_SSH,
    BLINE_KILLCHAIN_STANDARD,
)
from jaxmarl.environments.cage.actions import BLUE_SLEEP


requires_cyborg = pytest.mark.skipif(
    not is_cyborg_available(),
    reason="CybORG not installed"
)


@requires_cyborg
class TestScanImpliesDiscovery:
    """Scan (DiscoverNetworkServices) should mark the host as discovered."""

    def test_scan_without_discover_subnet_marks_discovered(self):
        """After scanning Enterprise1 (no prior DiscoverSubnet for Enterprise),
        both CybORG and JAX should report Enterprise1 as discovered."""
        harness = DifferentialHarness(seed=42, max_steps=10, verbose=False)

        red_actions = [
            red_discover_subnet(SUBNET_USER),
            red_scan_host('User1'),
            red_exploit_host('User1', EXPLOIT_SSH),
            red_privesc_host('User1'),
            red_scan_host('Enterprise1'),
        ]
        blue_actions = [BLUE_SLEEP] * len(red_actions)

        result = harness.run_episode(
            scripted_blue_policy_factory(blue_actions),
            scripted_red_policy_factory(red_actions),
        )

        step5 = result.step_results[4]
        assert step5.cyborg_state.red_discovered_hosts['Enterprise1'], \
            "CybORG should consider Enterprise1 discovered after scan"
        assert step5.jax_state.red_discovered_hosts['Enterprise1'], \
            "JAX should consider Enterprise1 discovered after scan"

    def test_scan_marks_discovered_for_user_host(self):
        """Scanning a User host (with DiscoverSubnet done) should also mark
        it discovered in both environments."""
        harness = DifferentialHarness(seed=42, max_steps=5, verbose=False)

        red_actions = [
            red_discover_subnet(SUBNET_USER),
            red_scan_host('User1'),
        ]
        blue_actions = [BLUE_SLEEP] * len(red_actions)

        result = harness.run_episode(
            scripted_blue_policy_factory(blue_actions),
            scripted_red_policy_factory(red_actions),
        )

        step2 = result.step_results[1]
        assert step2.jax_state.red_discovered_hosts['User1'], \
            "JAX should mark User1 discovered after scan"


@requires_cyborg
class TestExploitImpliesDiscovery:
    """Successful exploit should mark the host as discovered."""

    def test_exploit_marks_discovered(self):
        """After a successful exploit on User1, both should report discovered."""
        harness = DifferentialHarness(seed=42, max_steps=8, verbose=False)

        red_actions = [
            red_discover_subnet(SUBNET_USER),
            red_scan_host('User1'),
            red_exploit_host('User1', EXPLOIT_SSH),
        ]
        blue_actions = [BLUE_SLEEP] * len(red_actions)

        result = harness.run_episode(
            scripted_blue_policy_factory(blue_actions),
            scripted_red_policy_factory(red_actions),
        )

        step3 = result.step_results[2]
        assert step3.cyborg_state.red_discovered_hosts['User1'], \
            "CybORG should consider User1 discovered after exploit"
        assert step3.jax_state.red_discovered_hosts['User1'], \
            "JAX should consider User1 discovered after exploit"


@requires_cyborg
class TestBlineKillchainDiscovery:
    """The standard B_line killchain scans Enterprise1 at step 4 without
    DiscoverSubnet on Enterprise subnet. This should not produce a
    red_discovered_hosts mismatch."""

    def test_discovery_resolved_after_scan_in_bline_killchain(self):
        """After the full B_line killchain, scan/exploit-based discovery
        should eliminate persistent red_discovered_hosts warnings.

        CybORG reveals some hosts during PrivEsc (via observation side effects),
        so there may be a 1-step transient gap before scan sets discovered.
        The key invariant: once a host is scanned or exploited, both envs agree.
        """
        harness = DifferentialHarness(seed=42, max_steps=20, verbose=False)

        red_actions = BLINE_KILLCHAIN_STANDARD[:15]
        blue_actions = [BLUE_SLEEP] * len(red_actions)

        result = harness.run_episode(
            scripted_blue_policy_factory(blue_actions),
            scripted_red_policy_factory(red_actions),
        )

        last_step = result.step_results[-1]
        final_discovery_diffs = [
            d for d in last_step.diffs
            if d.field == 'red_discovered_hosts'
        ]
        assert len(final_discovery_diffs) == 0, (
            f"Expected no discovery diffs at final step, got: "
            + "; ".join(str(d) for d in final_discovery_diffs)
        )
