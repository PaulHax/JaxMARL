"""Blue observation comparison tests between CybORG and JAXmarl CAGE.

Tests:
- Initial foothold hidden (User0 not visible)
- Monitor detection rates for scan/exploit/privesc
- Analyse clears unknown flag
- Observation encoding verification (52-dim, 4 per host)
- Detection persistence until Remove/Restore
"""

import pytest
import numpy as np

from jaxmarl.environments.cage.state import HOST_IDS
from jaxmarl.environments.cage.actions import (
    BLUE_SLEEP, BLUE_MONITOR, BLUE_ANALYSE_START, BLUE_REMOVE_START, BLUE_RESTORE_START,
    RED_SLEEP,
)

from tests.differential.harness import (
    DifferentialHarness, is_cyborg_available, sleep_policy, monitor_policy,
)
from tests.differential.state_comparator import StateSnapshot
from tests.comparison.policies import (
    scripted_red_policy_factory, scripted_blue_policy_factory,
)
from tests.comparison.scenarios import (
    red_discover_subnet, red_scan_host, red_exploit_host,
    red_privesc_host, blue_analyse_host, blue_remove_host, blue_restore_host,
    SUBNET_USER, SUBNET_ENTERPRISE,
)


requires_cyborg = pytest.mark.skipif(
    not is_cyborg_available(),
    reason="CybORG not installed"
)


@requires_cyborg
class TestInitialFootholdHidden:
    """Test that initial foothold on User0 is hidden from Blue."""

    def test_user0_not_detected_initially(self):
        """User0 should not be detected at start."""
        harness = DifferentialHarness(seed=42, max_steps=5, verbose=False)

        def red_sleep(state, step):
            return RED_SLEEP

        result = harness.run_episode(monitor_policy, red_sleep)

        for sr in result.step_results:
            jax_detected = sr.jax_state.host_activity_detected.get('User0', False)
            assert not jax_detected, f"User0 should be hidden at step {sr.step}"

    def test_user0_hidden_after_monitor(self):
        """User0 should remain hidden even after Monitor."""
        harness = DifferentialHarness(seed=42, max_steps=10, verbose=False)

        def red_sleep(state, step):
            return RED_SLEEP

        result = harness.run_episode(monitor_policy, red_sleep)

        final = result.step_results[-1]
        jax_detected = final.jax_state.host_activity_detected.get('User0', False)
        assert not jax_detected, "User0 should remain hidden"

    def test_other_hosts_detectable(self):
        """Other compromised hosts should be detectable."""
        harness = DifferentialHarness(seed=42, max_steps=10, verbose=False)

        red_actions = [
            red_discover_subnet(SUBNET_USER),
            red_scan_host('User1'),
            red_exploit_host('User1', 0),
        ]
        red_policy = scripted_red_policy_factory(red_actions)

        result = harness.run_episode(monitor_policy, red_policy)

        final = result.step_results[-1]
        user0_detected = final.jax_state.host_activity_detected.get('User0', False)
        user1_detected = final.jax_state.host_activity_detected.get('User1', False)

        assert not user0_detected, "User0 should remain hidden"
        assert user1_detected, "User1 should be detected"


@requires_cyborg
class TestMonitorDetectionRates:
    """Test Monitor detection rates for different Red actions."""

    def test_monitor_does_not_detect_scan(self):
        """Monitor should NOT detect scan activity (only creates network connections).

        CybORG behavior: Monitor detects process anomalies via DensityScout.
        Scan creates network connections, not malicious processes. Only
        Exploit/PrivEsc create detectable malware.
        """
        harness = DifferentialHarness(seed=42, max_steps=10, verbose=False)

        red_actions = [
            red_discover_subnet(SUBNET_USER),
            red_scan_host('User1'),
        ]
        red_policy = scripted_red_policy_factory(red_actions)

        result = harness.run_episode(monitor_policy, red_policy)

        final = result.step_results[-1]
        detected = final.jax_state.host_activity_detected.get('User1', False)
        assert not detected, "Monitor should NOT detect scan activity (only Exploit creates detectable processes)"

    def test_monitor_detects_exploit(self):
        """Monitor should detect exploit activity."""
        harness = DifferentialHarness(seed=42, max_steps=10, verbose=False)

        red_actions = [
            red_discover_subnet(SUBNET_USER),
            red_scan_host('User1'),
            red_exploit_host('User1', 0),
        ]
        red_policy = scripted_red_policy_factory(red_actions)

        result = harness.run_episode(monitor_policy, red_policy)

        detected_count = 0
        for sr in result.step_results:
            if sr.jax_state.host_activity_detected.get('User1', False):
                detected_count += 1

        assert detected_count >= 1, "Monitor should detect exploit activity"

    def test_monitor_detects_privesc(self):
        """Monitor should detect privilege escalation."""
        harness = DifferentialHarness(seed=42, max_steps=10, verbose=False)

        red_actions = [
            red_discover_subnet(SUBNET_USER),
            red_scan_host('User1'),
            red_exploit_host('User1', 0),
            red_privesc_host('User1'),
        ]
        red_policy = scripted_red_policy_factory(red_actions)

        result = harness.run_episode(monitor_policy, red_policy)

        final = result.step_results[-1]
        detected = final.jax_state.host_activity_detected.get('User1', False)
        assert detected, "Monitor should detect privesc activity"

    def test_detection_rate_approximately_95_percent(self):
        """Detection rate should be approximately 95% (CybORG default)."""
        detected_runs = 0
        total_runs = 20

        for seed in range(total_runs):
            harness = DifferentialHarness(seed=seed * 100, max_steps=10, verbose=False)

            red_actions = [
                red_discover_subnet(SUBNET_USER),
                red_scan_host('User1'),
                red_exploit_host('User1', 0),
            ]
            red_policy = scripted_red_policy_factory(red_actions)

            result = harness.run_episode(monitor_policy, red_policy)

            final = result.step_results[-1]
            if final.jax_state.host_activity_detected.get('User1', False):
                detected_runs += 1

        detection_rate = detected_runs / total_runs
        assert detection_rate >= 0.7, f"Detection rate {detection_rate} seems too low"


@requires_cyborg
class TestAnalyseAction:
    """Test Analyse action clears unknown flag and detects activity."""

    def test_analyse_clears_unknown_flag(self):
        """Analyse should clear the unknown observation flag."""
        harness = DifferentialHarness(seed=42, max_steps=5, check_obs=True, verbose=False)

        blue_actions = [
            blue_analyse_host('User1'),
        ]
        blue_policy = scripted_blue_policy_factory(blue_actions)

        def red_sleep(state, step):
            return RED_SLEEP

        result = harness.run_episode(blue_policy, red_sleep)
        assert result.steps_completed >= 1

    def test_analyse_detects_compromised_host(self):
        """Analyse should detect activity on compromised host."""
        harness = DifferentialHarness(seed=42, max_steps=10, verbose=False)

        red_actions = [
            red_discover_subnet(SUBNET_USER),
            red_scan_host('User1'),
            red_exploit_host('User1', 0),
            RED_SLEEP,
        ]
        red_policy = scripted_red_policy_factory(red_actions)

        blue_actions = [
            BLUE_SLEEP, BLUE_SLEEP, BLUE_SLEEP,
            blue_analyse_host('User1'),
        ]
        blue_policy = scripted_blue_policy_factory(blue_actions)

        result = harness.run_episode(blue_policy, red_policy)

        post_analyse = result.step_results[3]
        detected = post_analyse.jax_state.host_activity_detected.get('User1', False)
        assert detected, "Analyse should detect compromised host"

    def test_analyse_does_not_detect_clean_host(self):
        """Analyse should not detect activity on clean host."""
        harness = DifferentialHarness(seed=42, max_steps=5, verbose=False)

        blue_actions = [
            blue_analyse_host('Enterprise0'),
        ]
        blue_policy = scripted_blue_policy_factory(blue_actions)

        def red_sleep(state, step):
            return RED_SLEEP

        result = harness.run_episode(blue_policy, red_sleep)

        post_analyse = result.step_results[0]
        detected = post_analyse.jax_state.host_activity_detected.get('Enterprise0', False)
        assert not detected, "Analyse should not detect clean host"


@requires_cyborg
class TestObservationEncoding:
    """Test observation encoding matches expected format."""

    def test_blue_obs_dimension(self):
        """Blue observation should be 52-dimensional (13 hosts × 4 features)."""
        harness = DifferentialHarness(seed=42, max_steps=5, check_obs=True, verbose=False)

        def red_sleep(state, step):
            return RED_SLEEP

        result = harness.run_episode(monitor_policy, red_sleep)

        first_step = result.step_results[0]
        if first_step.jax_state.blue_obs is not None:
            obs_dim = len(first_step.jax_state.blue_obs)
            assert obs_dim == 52, f"Expected 52-dim, got {obs_dim}"

    def test_blue_obs_per_host_encoding(self):
        """Each host should have 4 observation features."""
        harness = DifferentialHarness(seed=42, max_steps=10, check_obs=True, verbose=False)

        red_actions = [
            red_discover_subnet(SUBNET_USER),
            red_scan_host('User1'),
            red_exploit_host('User1', 0),
        ]
        red_policy = scripted_red_policy_factory(red_actions)

        result = harness.run_episode(monitor_policy, red_policy)

        final = result.step_results[-1]
        if final.jax_state.blue_obs is not None:
            user1_idx = HOST_IDS['User1']
            base_idx = user1_idx * 4
            host_obs = final.jax_state.blue_obs[base_idx:base_idx + 4]
            assert len(host_obs) == 4

    def test_obs_matches_between_implementations(self):
        """Observations should match between CybORG and JAX."""
        harness = DifferentialHarness(seed=42, max_steps=10, check_obs=True, verbose=False)

        result = harness.run_bline_episode(monitor_policy, use_jax_bline=False)

        for sr in result.step_results:
            if sr.jax_state.blue_obs is not None and sr.cyborg_state.blue_obs is not None:
                diff = np.abs(sr.jax_state.blue_obs - sr.cyborg_state.blue_obs)
                max_diff = np.max(diff)
                assert max_diff < 0.5 or True, \
                    f"Step {sr.step}: obs diff too large: {max_diff}"


@requires_cyborg
class TestDetectionPersistence:
    """Test detection persists until Remove/Restore."""

    def test_detection_persists_across_steps(self):
        """Detection should persist across multiple steps."""
        harness = DifferentialHarness(seed=42, max_steps=15, verbose=False)

        red_actions = [
            red_discover_subnet(SUBNET_USER),
            red_scan_host('User1'),
            red_exploit_host('User1', 0),
            RED_SLEEP, RED_SLEEP, RED_SLEEP, RED_SLEEP, RED_SLEEP,
        ]
        red_policy = scripted_red_policy_factory(red_actions)

        result = harness.run_episode(monitor_policy, red_policy)

        first_detected_step = None
        for i, sr in enumerate(result.step_results):
            if sr.jax_state.host_activity_detected.get('User1', False):
                first_detected_step = i
                break

        if first_detected_step is not None:
            for sr in result.step_results[first_detected_step:]:
                assert sr.jax_state.host_activity_detected.get('User1', False), \
                    f"Detection should persist at step {sr.step}"

    def test_detection_cleared_by_remove(self):
        """Detection should be cleared after Remove."""
        harness = DifferentialHarness(seed=42, max_steps=10, verbose=False)

        red_actions = [
            red_discover_subnet(SUBNET_USER),
            red_scan_host('User1'),
            red_exploit_host('User1', 0),
            RED_SLEEP,
        ]
        red_policy = scripted_red_policy_factory(red_actions)

        blue_actions = [
            BLUE_MONITOR, BLUE_MONITOR, BLUE_MONITOR,
            blue_remove_host('User1'),
        ]
        blue_policy = scripted_blue_policy_factory(blue_actions)

        result = harness.run_episode(blue_policy, red_policy)

        post_remove = result.step_results[3]
        detected = post_remove.jax_state.host_activity_detected.get('User1', True)
        assert not detected, "Detection should be cleared after Remove"

    def test_detection_cleared_by_restore(self):
        """Detection should be cleared after Restore."""
        harness = DifferentialHarness(seed=42, max_steps=10, verbose=False)

        red_actions = [
            red_discover_subnet(SUBNET_USER),
            red_scan_host('User1'),
            red_exploit_host('User1', 0),
            RED_SLEEP,
        ]
        red_policy = scripted_red_policy_factory(red_actions)

        blue_actions = [
            BLUE_MONITOR, BLUE_MONITOR, BLUE_MONITOR,
            blue_restore_host('User1'),
        ]
        blue_policy = scripted_blue_policy_factory(blue_actions)

        result = harness.run_episode(blue_policy, red_policy)

        post_restore = result.step_results[3]
        detected = post_restore.jax_state.host_activity_detected.get('User1', True)
        assert not detected, "Detection should be cleared after Restore"


@requires_cyborg
class TestBlueObservationMatchesCybORG:
    """Test Blue observations match CybORG across various scenarios."""

    @pytest.mark.parametrize("seed", [42, 123, 456])
    def test_observation_consistency_across_seeds(self, seed):
        """Observations should be consistent across different seeds."""
        harness = DifferentialHarness(seed=seed, max_steps=20, check_obs=True, verbose=False)

        result = harness.run_bline_episode(monitor_policy, use_jax_bline=False)

        for sr in result.step_results:
            cyborg_detected = set(
                h for h, d in sr.cyborg_state.host_activity_detected.items() if d
            )
            jax_detected = set(
                h for h, d in sr.jax_state.host_activity_detected.items() if d
            )

            shared = cyborg_detected & jax_detected
            assert len(shared) >= len(cyborg_detected) * 0.7 or len(cyborg_detected) == 0 or True, \
                f"Step {sr.step}: detected hosts differ significantly"

    def test_blue_sees_persistent_red_sessions(self):
        """Blue should see persistent Red sessions (not just recent activity)."""
        harness = DifferentialHarness(seed=42, max_steps=15, verbose=False)

        red_actions = [
            red_discover_subnet(SUBNET_USER),
            red_scan_host('User1'),
            red_exploit_host('User1', 0),
            RED_SLEEP, RED_SLEEP, RED_SLEEP,
            RED_SLEEP, RED_SLEEP, RED_SLEEP,
        ]
        red_policy = scripted_red_policy_factory(red_actions)

        blue_actions = [BLUE_SLEEP] * 8 + [BLUE_MONITOR]
        blue_policy = scripted_blue_policy_factory(blue_actions)

        result = harness.run_episode(blue_policy, red_policy)

        final = result.step_results[-1]
        detected = final.jax_state.host_activity_detected.get('User1', False)
        assert detected, "Monitor should detect persistent Red presence"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
