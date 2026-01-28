"""State transition comparison tests between CybORG and JAXmarl CAGE.

Tests:
- Compromise level transitions: NONE→USER→PRIVILEGED→NONE
- Session count tracking through exploit/remove/restore
- Discovery and scan state persistence
- OT service state after Impact/Restore
"""

import pytest
import numpy as np

from jaxmarl.environments.cage.state import (
    HOST_IDS, COMPROMISE_NONE, COMPROMISE_USER, COMPROMISE_PRIVILEGED,
)
from jaxmarl.environments.cage.actions import BLUE_SLEEP, BLUE_MONITOR, RED_SLEEP

from tests.differential.harness import (
    DifferentialHarness, is_cyborg_available, sleep_policy, monitor_policy,
)
from tests.differential.state_comparator import StateSnapshot
from tests.comparison.policies import scripted_blue_policy_factory
from tests.comparison.scenarios import blue_restore_host


requires_cyborg = pytest.mark.skipif(
    not is_cyborg_available(),
    reason="CybORG not installed"
)


@requires_cyborg
class TestCompromiseLevelTransitions:
    """Test compromise level transitions match CybORG."""

    def test_compromise_progression_in_bline(self):
        """Compromise levels should progress NONE→USER→PRIVILEGED."""
        harness = DifferentialHarness(seed=42, max_steps=20, verbose=False)
        result = harness.run_bline_episode(sleep_policy, use_jax_bline=False)

        user_compromised = False
        privileged_reached = False

        for sr in result.step_results:
            for hostname in HOST_IDS:
                if hostname == 'User0':
                    continue

                jax_priv = sr.jax_state.red_privilege.get(hostname, 0)
                cyborg_priv = sr.cyborg_state.red_privilege.get(hostname, 0)

                assert jax_priv == cyborg_priv, \
                    f"Step {sr.step}: {hostname} privilege mismatch JAX={jax_priv} CybORG={cyborg_priv}"

                if jax_priv == COMPROMISE_USER:
                    user_compromised = True
                if jax_priv == COMPROMISE_PRIVILEGED:
                    privileged_reached = True

        assert user_compromised, "Should reach USER compromise level"
        assert privileged_reached, "Should reach PRIVILEGED level"

    def test_restore_clears_privilege(self):
        """Restore should clear privilege level."""
        harness = DifferentialHarness(seed=42, max_steps=25, verbose=False)

        def blue_restore_at_step_18(state: StateSnapshot, step: int) -> int:
            if step == 18:
                return blue_restore_host('Enterprise1')
            return BLUE_SLEEP

        result = harness.run_bline_episode(blue_restore_at_step_18, use_jax_bline=False)

        assert result.error_diffs == 0


@requires_cyborg
class TestSessionCountTracking:
    """Test session count tracking matches CybORG."""

    def test_sessions_match_throughout_episode(self):
        """Session counts should match at each step."""
        harness = DifferentialHarness(seed=42, max_steps=20, verbose=False)
        result = harness.run_bline_episode(sleep_policy, use_jax_bline=False)

        for sr in result.step_results:
            for hostname in HOST_IDS:
                jax_sessions = sr.jax_state.red_sessions.get(hostname, 0)
                cyborg_sessions = sr.cyborg_state.red_sessions.get(hostname, 0)

                jax_has = jax_sessions > 0
                cyborg_has = cyborg_sessions > 0

                assert jax_has == cyborg_has, \
                    f"Step {sr.step}: {hostname} session presence mismatch"

    def test_sessions_accumulate_on_exploits(self):
        """Sessions should increase on successful exploits."""
        harness = DifferentialHarness(seed=42, max_steps=20, verbose=False)
        result = harness.run_bline_episode(sleep_policy, use_jax_bline=False)

        total_sessions_over_time = []
        for sr in result.step_results:
            total = sum(sr.jax_state.red_sessions.values())
            total_sessions_over_time.append(total)

        assert max(total_sessions_over_time) >= 4, \
            f"Should accumulate sessions, max was {max(total_sessions_over_time)}"


@requires_cyborg
class TestDiscoveryScanPersistence:
    """Test discovery and scan state persistence."""

    def test_discovery_state_tracked(self):
        """Discovery state should be tracked consistently."""
        harness = DifferentialHarness(seed=42, max_steps=20, verbose=False)
        result = harness.run_bline_episode(sleep_policy, use_jax_bline=False)

        final = result.step_results[-1]
        discovered_count = sum(1 for v in final.jax_state.red_discovered_hosts.values() if v)

        assert discovered_count >= 5, \
            f"Should discover multiple hosts, got {discovered_count}"

    def test_scan_state_tracked(self):
        """Scan state should be tracked consistently."""
        harness = DifferentialHarness(seed=42, max_steps=20, verbose=False)
        result = harness.run_bline_episode(sleep_policy, use_jax_bline=False)

        final = result.step_results[-1]
        scanned_count = sum(1 for v in final.jax_state.red_scanned_hosts.values() if v)

        assert scanned_count >= 4, \
            f"Should scan multiple hosts, got {scanned_count}"


@requires_cyborg
class TestOTServiceState:
    """Test OT service state after Impact/Restore."""

    def test_impact_stops_ot_service(self):
        """Impact should stop OT service."""
        harness = DifferentialHarness(seed=42, max_steps=20, verbose=False)
        result = harness.run_bline_episode(sleep_policy, use_jax_bline=False)

        impact_found = False
        for sr in result.step_results:
            if 'Impact' in sr.red_action_desc:
                impact_found = True
                assert sr.jax_state.ot_service_stopped.get('Op_Server0', False), \
                    "Impact should stop OT service"
                break

        assert impact_found, "Should reach Impact action"

    def test_restore_restarts_ot_service(self):
        """Restore should restart OT service."""
        harness = DifferentialHarness(seed=42, max_steps=25, verbose=False)

        def blue_restore_at_step_20(state: StateSnapshot, step: int) -> int:
            if step == 20:
                return blue_restore_host('Op_Server0')
            return BLUE_SLEEP

        result = harness.run_bline_episode(blue_restore_at_step_20, use_jax_bline=False)

        if len(result.step_results) > 20:
            sr = result.step_results[20]
            ot_stopped = sr.jax_state.ot_service_stopped.get('Op_Server0', True)
            assert not ot_stopped, "Restore should restart OT service"


@requires_cyborg
class TestActivityDetectionState:
    """Test activity detection state transitions."""

    def test_monitor_detects_activity(self):
        """Monitor should detect Red activity."""
        harness = DifferentialHarness(seed=42, max_steps=20, verbose=False)
        result = harness.run_bline_episode(monitor_policy, use_jax_bline=False)

        detected_any = False
        for sr in result.step_results:
            for hostname, detected in sr.jax_state.host_activity_detected.items():
                if detected and hostname != 'User0':
                    detected_any = True
                    break

        assert detected_any, "Monitor should detect some activity"

    def test_detection_persists(self):
        """Detection should persist until cleared."""
        harness = DifferentialHarness(seed=42, max_steps=20, verbose=False)
        result = harness.run_bline_episode(monitor_policy, use_jax_bline=False)

        first_detection_step = {}
        for i, sr in enumerate(result.step_results):
            for hostname, detected in sr.jax_state.host_activity_detected.items():
                if detected and hostname not in first_detection_step:
                    first_detection_step[hostname] = i

        for hostname, first_step in first_detection_step.items():
            for sr in result.step_results[first_step:]:
                if hostname != 'User0':
                    assert sr.jax_state.host_activity_detected.get(hostname, False), \
                        f"Detection on {hostname} should persist"


@requires_cyborg
class TestStateConsistencyAcrossSeeds:
    """Test state consistency across different seeds."""

    @pytest.mark.parametrize("seed", [42, 123, 456])
    def test_compromise_state_matches(self, seed):
        """Compromise state should match across seeds."""
        harness = DifferentialHarness(seed=seed, max_steps=30, verbose=False)
        result = harness.run_bline_episode(sleep_policy, use_jax_bline=False)

        for sr in result.step_results:
            for hostname in HOST_IDS:
                jax_comp = sr.jax_state.host_compromised.get(hostname, 0)
                cyborg_comp = sr.cyborg_state.host_compromised.get(hostname, 0)

                assert jax_comp == cyborg_comp, \
                    f"Seed {seed}, Step {sr.step}: {hostname} compromise mismatch"

    @pytest.mark.parametrize("seed", [42, 123, 456])
    def test_privilege_state_matches(self, seed):
        """Privilege state should match across seeds."""
        harness = DifferentialHarness(seed=seed, max_steps=30, verbose=False)
        result = harness.run_bline_episode(sleep_policy, use_jax_bline=False)

        for sr in result.step_results:
            for hostname in HOST_IDS:
                jax_priv = sr.jax_state.red_privilege.get(hostname, 0)
                cyborg_priv = sr.cyborg_state.red_privilege.get(hostname, 0)

                assert jax_priv == cyborg_priv, \
                    f"Seed {seed}, Step {sr.step}: {hostname} privilege mismatch"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
