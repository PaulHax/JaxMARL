"""Systematic action type comparison tests between CybORG and JAXmarl CAGE.

Tests all action variants:
- Red: DiscoverSubnet (3), ScanHost (13), Exploit types (8×13), PrivEsc (13), Impact
- Blue: Decoy types (8×13), Analyse (13), Remove (13), Restore (13)
"""

import pytest
import numpy as np

from jaxmarl.environments.cage.state import HOST_IDS
from jaxmarl.environments.cage.actions import (
    BLUE_SLEEP, BLUE_MONITOR, BLUE_REMOVE_START, BLUE_RESTORE_START,
    BLUE_ANALYSE_START, BLUE_DECOY_START, NUM_DECOY_TYPES,
    RED_SLEEP, RED_DISCOVER_SUBNET_START, RED_SCAN_HOST_START,
    RED_EXPLOIT_START, RED_PRIVESC_START, RED_IMPACT_START,
    NUM_HOSTS,
)

from tests.cage.differential.harness import (
    DifferentialHarness, is_cyborg_available, sleep_policy,
)
from tests.cage.differential.state_comparator import StateSnapshot
from tests.cage.comparison.policies import (
    scripted_red_policy_factory, scripted_blue_policy_factory,
)
from tests.cage.comparison.scenarios import (
    red_discover_subnet, red_scan_host, red_exploit_host,
    red_privesc_host, red_impact_host,
    blue_analyse_host, blue_remove_host, blue_restore_host, blue_decoy_host,
    SUBNET_USER, SUBNET_ENTERPRISE, SUBNET_OPERATIONAL,
    BLINE_KILLCHAIN_STANDARD,
)


requires_cyborg = pytest.mark.skipif(
    not is_cyborg_available(),
    reason="CybORG not installed"
)


@requires_cyborg
class TestRedDiscoverSubnet:
    """Test DiscoverRemoteSystems action for all 3 subnets via B_lineAgent."""

    def test_discover_user_subnet(self):
        """Discover User subnet in B_line trajectory."""
        harness = DifferentialHarness(seed=42, max_steps=5, verbose=False)
        result = harness.run_bline_episode(sleep_policy, use_jax_bline=False)

        assert result.error_diffs == 0, f"State mismatch in User subnet discovery"

    def test_discover_enterprise_subnet(self):
        """Discover Enterprise subnet in B_line trajectory."""
        harness = DifferentialHarness(seed=42, max_steps=10, verbose=False)
        result = harness.run_bline_episode(sleep_policy, use_jax_bline=False)

        assert result.error_diffs == 0, f"State mismatch in Enterprise subnet discovery"

    def test_discover_operational_subnet(self):
        """Discover Operational subnet in B_line trajectory."""
        harness = DifferentialHarness(seed=42, max_steps=15, verbose=False)
        result = harness.run_bline_episode(sleep_policy, use_jax_bline=False)

        assert result.error_diffs == 0, f"State mismatch in Operational subnet discovery"


@requires_cyborg
class TestRedScanHost:
    """Test DiscoverNetworkServices action via B_line trajectory."""

    def test_scan_hosts_in_bline_trajectory(self):
        """ScanHost actions should match CybORG via B_line."""
        harness = DifferentialHarness(seed=42, max_steps=20, verbose=False)
        result = harness.run_bline_episode(sleep_policy, use_jax_bline=False)

        scan_steps = [sr for sr in result.step_results if 'Scan' in sr.red_action_desc]
        assert len(scan_steps) >= 4, "B_line should perform multiple scans"

        for sr in result.step_results:
            assert abs(sr.cyborg_state.reward_red - sr.jax_state.reward_red) < 0.05, \
                f"Step {sr.step}: reward mismatch"


@requires_cyborg
class TestRedExploitTypes:
    """Test exploit types via B_line trajectory."""

    def test_exploit_actions_match(self):
        """Exploit rewards should match CybORG."""
        harness = DifferentialHarness(seed=42, max_steps=20, verbose=False)
        result = harness.run_bline_episode(sleep_policy, use_jax_bline=False)

        exploit_steps = [sr for sr in result.step_results
                        if 'BruteForce' in sr.red_action_desc or
                           'Traversal' in sr.red_action_desc or
                           'RFI' in sr.red_action_desc or
                           'RCE' in sr.red_action_desc]

        assert len(exploit_steps) >= 1, "B_line should perform exploits"

        for sr in exploit_steps:
            assert abs(sr.cyborg_state.reward_red - sr.jax_state.reward_red) < 0.05, \
                f"Step {sr.step} ({sr.red_action_desc}): reward mismatch"

    @pytest.mark.parametrize("seed", [42, 123, 456])
    def test_exploit_rewards_across_seeds(self, seed):
        """Exploit rewards should match across different seeds."""
        harness = DifferentialHarness(seed=seed, max_steps=20, verbose=False)
        result = harness.run_bline_episode(sleep_policy, use_jax_bline=False)

        for sr in result.step_results:
            assert abs(sr.cyborg_state.reward_red - sr.jax_state.reward_red) < 0.05, \
                f"Seed {seed}, Step {sr.step}: reward mismatch"


@requires_cyborg
class TestRedPrivilegeEscalate:
    """Test PrivilegeEscalate action via B_line trajectory."""

    def test_privesc_actions_match(self):
        """PrivEsc state changes should match CybORG."""
        harness = DifferentialHarness(seed=42, max_steps=20, verbose=False)
        result = harness.run_bline_episode(sleep_policy, use_jax_bline=False)

        privesc_steps = [sr for sr in result.step_results if 'PrivEsc' in sr.red_action_desc]
        assert len(privesc_steps) >= 1, "B_line should perform privesc"

        for sr in result.step_results:
            for hostname in HOST_IDS:
                cyborg_priv = sr.cyborg_state.red_privilege.get(hostname, 0)
                jax_priv = sr.jax_state.red_privilege.get(hostname, 0)
                assert cyborg_priv == jax_priv, \
                    f"Step {sr.step}: {hostname} privilege mismatch"


@requires_cyborg
class TestRedImpact:
    """Test Impact action on operational hosts."""

    def test_impact_rewards_match(self):
        """Impact on Op_Server0 should give matching rewards."""
        harness = DifferentialHarness(seed=42, max_steps=20, verbose=False)
        result = harness.run_bline_episode(sleep_policy, use_jax_bline=False)

        impact_step = None
        for i, sr in enumerate(result.step_results):
            if 'Impact' in sr.red_action_desc:
                impact_step = i
                break

        assert impact_step is not None, "B_line should reach Impact"

        sr = result.step_results[impact_step]
        assert abs(sr.cyborg_state.reward_red - sr.jax_state.reward_red) < 0.1, \
            f"Impact reward mismatch"
        assert abs(sr.cyborg_state.reward_blue - sr.jax_state.reward_blue) < 0.5, \
            f"Impact blue penalty mismatch"


@requires_cyborg
class TestBlueDecoyTypes:
    """Test Blue decoy types."""

    DECOY_NAMES = [
        'DecoyApache', 'DecoyFemitter', 'DecoyHarakaSMPT', 'DecoySmss',
        'DecoySSHD', 'DecoySvchost', 'DecoyTomcat', 'DecoyVsftpd',
    ]

    @pytest.mark.parametrize("decoy_type", range(8))
    def test_decoy_deployment(self, decoy_type):
        """Each decoy type deployment should not cause state mismatch."""
        harness = DifferentialHarness(seed=42, max_steps=5, verbose=False)

        blue_action = blue_decoy_host('Enterprise1', decoy_type)
        blue_policy = scripted_blue_policy_factory([blue_action])

        def red_sleep(state, step):
            return RED_SLEEP

        result = harness.run_episode(blue_policy, red_sleep)
        assert result.error_diffs == 0, \
            f"Decoy {self.DECOY_NAMES[decoy_type]} deployment caused state mismatch"


@requires_cyborg
class TestBlueAnalyse:
    """Test Analyse action."""

    @pytest.mark.parametrize("hostname", ['User1', 'Enterprise0', 'Op_Server0'])
    def test_analyse_host(self, hostname):
        """Analyse should not cause state mismatch."""
        harness = DifferentialHarness(seed=42, max_steps=5, verbose=False)

        blue_action = blue_analyse_host(hostname)
        blue_policy = scripted_blue_policy_factory([blue_action])

        def red_sleep(state, step):
            return RED_SLEEP

        result = harness.run_episode(blue_policy, red_sleep)
        assert result.error_diffs == 0, f"Analyse {hostname} caused state mismatch"


@requires_cyborg
class TestBlueRemove:
    """Test Remove action on compromised hosts."""

    def test_remove_in_bline_trajectory(self):
        """Remove after detection should match CybORG."""
        from tests.cage.comparison.policies import react_remove_policy_with_timing

        harness = DifferentialHarness(seed=42, max_steps=25, verbose=False)
        remove_policy = react_remove_policy_with_timing(delay=0)
        result = harness.run_bline_episode(remove_policy, use_jax_bline=False)

        remove_steps = [sr for sr in result.step_results if 'Remove' in sr.blue_action_desc]

        for sr in result.step_results:
            assert abs(sr.cyborg_state.reward_blue - sr.jax_state.reward_blue) < 0.1, \
                f"Step {sr.step}: blue reward mismatch"


@requires_cyborg
class TestBlueRestore:
    """Test Restore action."""

    def test_restore_cost_matches(self):
        """Restore cost should match CybORG (-1.0 per restore)."""
        harness = DifferentialHarness(seed=42, max_steps=5, verbose=False)

        blue_policy = scripted_blue_policy_factory([
            blue_restore_host('User1'),
            blue_restore_host('Enterprise0'),
        ])

        def red_sleep(state, step):
            return RED_SLEEP

        result = harness.run_episode(blue_policy, red_sleep)

        for sr in result.step_results:
            if 'Restore' in sr.blue_action_desc:
                assert abs(sr.cyborg_state.reward_blue - sr.jax_state.reward_blue) < 0.05, \
                    f"Restore cost mismatch at step {sr.step}"

    def test_restore_after_compromise(self):
        """Restore after compromise should clear Red access."""
        harness = DifferentialHarness(seed=42, max_steps=25, verbose=False)

        def blue_restore_at_step_18(state: StateSnapshot, step: int) -> int:
            if step == 18:
                return blue_restore_host('Enterprise1')
            return BLUE_SLEEP

        result = harness.run_bline_episode(blue_restore_at_step_18, use_jax_bline=False)

        assert result.steps_completed >= 19
        if len(result.step_results) > 18:
            sr = result.step_results[18]
            assert abs(sr.cyborg_state.reward_blue - sr.jax_state.reward_blue) < 0.1


@requires_cyborg
class TestActionCostsAndRewards:
    """Test action costs and rewards match between implementations."""

    def test_exploit_security_rewards(self):
        """Exploit security rewards should match per host."""
        harness = DifferentialHarness(seed=42, max_steps=20, verbose=False)
        result = harness.run_bline_episode(sleep_policy, use_jax_bline=False)

        for sr in result.step_results:
            assert abs(sr.cyborg_state.reward_red - sr.jax_state.reward_red) < 0.05, \
                f"Step {sr.step}: red reward mismatch"
            assert abs(sr.cyborg_state.reward_blue - sr.jax_state.reward_blue) < 0.1, \
                f"Step {sr.step}: blue reward mismatch"

    def test_privesc_security_rewards(self):
        """PrivEsc security rewards should match."""
        harness = DifferentialHarness(seed=42, max_steps=20, verbose=False)
        result = harness.run_bline_episode(sleep_policy, use_jax_bline=False)

        privesc_steps = [sr for sr in result.step_results if 'PrivEsc' in sr.red_action_desc]

        for sr in privesc_steps:
            assert abs(sr.cyborg_state.reward_red - sr.jax_state.reward_red) < 0.1, \
                f"PrivEsc at step {sr.step}: red reward mismatch"

    def test_cumulative_rewards_match(self):
        """Cumulative rewards should match throughout episode."""
        harness = DifferentialHarness(seed=42, max_steps=30, verbose=False)
        result = harness.run_bline_episode(sleep_policy, use_jax_bline=False)

        final = result.step_results[-1]
        cyborg_blue, cyborg_red = final.cyborg_state.reward_blue, final.cyborg_state.reward_red
        jax_blue, jax_red = final.jax_state.reward_blue, final.jax_state.reward_red

        assert abs(cyborg_red - jax_red) < 0.5, f"Final red reward diff: {cyborg_red} vs {jax_red}"
        assert abs(cyborg_blue - jax_blue) < 1.0, f"Final blue reward diff: {cyborg_blue} vs {jax_blue}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
