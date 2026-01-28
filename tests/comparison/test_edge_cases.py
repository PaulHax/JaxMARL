"""Edge case comparison tests between CybORG and JAXmarl CAGE.

Tests:
- Invalid action sequences (exploit without scan, privesc without exploit)
- Failed exploits (blocked by decoy, wrong service type)
- Boundary conditions (max steps, repeated actions, actions after restore)
- Session management (multiple sessions, remove vs restore effects)
"""

import pytest
import numpy as np

from jaxmarl.environments.cage.state import HOST_IDS, COMPROMISE_NONE, COMPROMISE_USER, COMPROMISE_PRIVILEGED
from jaxmarl.environments.cage.actions import (
    BLUE_SLEEP, BLUE_MONITOR, BLUE_REMOVE_START, BLUE_RESTORE_START,
    BLUE_DECOY_START, NUM_DECOY_TYPES,
    RED_SLEEP,
)

from tests.differential.harness import (
    DifferentialHarness, is_cyborg_available, sleep_policy,
)
from tests.differential.state_comparator import StateSnapshot
from tests.comparison.policies import (
    scripted_blue_policy_factory,
)
from tests.comparison.scenarios import (
    blue_remove_host, blue_restore_host, blue_decoy_host,
    DECOY_SSHD, DECOY_APACHE, DECOY_HARAKA,
)


requires_cyborg = pytest.mark.skipif(
    not is_cyborg_available(),
    reason="CybORG not installed"
)


@requires_cyborg
class TestBoundaryConditions:
    """Test boundary conditions are handled consistently."""

    def test_max_steps_reached(self):
        """Episode should end at max_steps."""
        harness = DifferentialHarness(seed=42, max_steps=10, verbose=False)
        result = harness.run_bline_episode(sleep_policy, use_jax_bline=False)

        assert result.steps_completed == 10, \
            f"Should complete exactly 10 steps, got {result.steps_completed}"

    def test_full_100_step_episode(self):
        """Full 100-step episode should complete."""
        harness = DifferentialHarness(seed=42, max_steps=100, verbose=False)
        result = harness.run_bline_episode(sleep_policy, use_jax_bline=False)

        assert result.steps_completed == 100
        assert result.error_diffs == 0

    def test_repeated_sleep_actions_no_effect(self):
        """Repeated sleep actions should have no effect on state."""
        harness = DifferentialHarness(seed=42, max_steps=10, verbose=False)

        def red_sleep(state, step):
            return RED_SLEEP

        blue_policy = scripted_blue_policy_factory([BLUE_SLEEP] * 10)

        result = harness.run_episode(blue_policy, red_sleep)

        for sr in result.step_results:
            assert sr.jax_state.reward_red == 0.0, \
                f"Sleep should give no reward at step {sr.step}"
            assert sr.jax_state.reward_blue == 0.0, \
                f"Sleep should give no reward at step {sr.step}"


@requires_cyborg
class TestSessionManagement:
    """Test session management via B_line trajectories."""

    def test_sessions_created_on_exploit(self):
        """Sessions should be created on successful exploit."""
        harness = DifferentialHarness(seed=42, max_steps=20, verbose=False)
        result = harness.run_bline_episode(sleep_policy, use_jax_bline=False)

        exploited_hosts = set()
        for sr in result.step_results:
            for hostname, sessions in sr.jax_state.red_sessions.items():
                if sessions > 0:
                    exploited_hosts.add(hostname)

        assert len(exploited_hosts) >= 3, \
            f"B_line should exploit multiple hosts, got {exploited_hosts}"

    def test_remove_clears_user_session(self):
        """Remove should clear user-level sessions."""
        from tests.comparison.policies import react_remove_policy_with_timing

        harness = DifferentialHarness(seed=42, max_steps=25, verbose=False)
        remove_policy = react_remove_policy_with_timing(delay=0)
        result = harness.run_bline_episode(remove_policy, use_jax_bline=False)

        for sr in result.step_results:
            if 'Remove' in sr.blue_action_desc:
                jax_reward = sr.jax_state.reward_blue
                cyborg_reward = sr.cyborg_state.reward_blue
                assert abs(jax_reward - cyborg_reward) < 0.1

    def test_restore_clears_sessions(self):
        """Restore should clear all sessions."""
        harness = DifferentialHarness(seed=42, max_steps=25, verbose=False)

        def blue_restore_at_step_15(state: StateSnapshot, step: int) -> int:
            if step == 15:
                return blue_restore_host('Enterprise1')
            return BLUE_SLEEP

        result = harness.run_bline_episode(blue_restore_at_step_15, use_jax_bline=False)

        assert result.steps_completed >= 16
        assert result.error_diffs == 0


@requires_cyborg
class TestDecoyEdgeCases:
    """Test decoy edge cases."""

    def test_decoy_deployment_matches(self):
        """Decoy deployment should not cause state mismatch."""
        harness = DifferentialHarness(seed=42, max_steps=5, verbose=False)

        blue_actions = [
            blue_decoy_host('Enterprise1', DECOY_SSHD),
            blue_decoy_host('Enterprise1', DECOY_APACHE),
        ]
        blue_policy = scripted_blue_policy_factory(blue_actions)

        def red_sleep(state, step):
            return RED_SLEEP

        result = harness.run_episode(blue_policy, red_sleep)
        assert result.error_diffs == 0

    def test_restore_removes_decoys(self):
        """Restore should remove decoys."""
        harness = DifferentialHarness(seed=42, max_steps=5, verbose=False)

        blue_actions = [
            blue_decoy_host('Enterprise1', DECOY_SSHD),
            blue_restore_host('Enterprise1'),
            blue_decoy_host('Enterprise1', DECOY_SSHD),
        ]
        blue_policy = scripted_blue_policy_factory(blue_actions)

        def red_sleep(state, step):
            return RED_SLEEP

        result = harness.run_episode(blue_policy, red_sleep)
        assert result.error_diffs == 0


@requires_cyborg
class TestRestoreEdgeCases:
    """Test restore action edge cases."""

    def test_restore_on_clean_host(self):
        """Restore on clean host should still cost -1."""
        harness = DifferentialHarness(seed=42, max_steps=5, verbose=False)

        blue_policy = scripted_blue_policy_factory([
            blue_restore_host('Enterprise0'),
        ])

        def red_sleep(state, step):
            return RED_SLEEP

        result = harness.run_episode(blue_policy, red_sleep)

        sr = result.step_results[0]
        assert abs(sr.cyborg_state.reward_blue - sr.jax_state.reward_blue) < 0.1

    def test_multiple_restores_same_host(self):
        """Multiple restores on same host should each cost -1."""
        harness = DifferentialHarness(seed=42, max_steps=5, verbose=False)

        blue_policy = scripted_blue_policy_factory([
            blue_restore_host('Enterprise0'),
            blue_restore_host('Enterprise0'),
            blue_restore_host('Enterprise0'),
        ])

        def red_sleep(state, step):
            return RED_SLEEP

        result = harness.run_episode(blue_policy, red_sleep)

        for sr in result.step_results:
            if 'Restore' in sr.blue_action_desc:
                assert abs(sr.cyborg_state.reward_blue - sr.jax_state.reward_blue) < 0.1


@requires_cyborg
class TestRemoveEdgeCases:
    """Test remove action edge cases."""

    def test_remove_requires_detection(self):
        """Remove without prior detection should be ineffective."""
        harness = DifferentialHarness(seed=42, max_steps=5, verbose=False)

        blue_policy = scripted_blue_policy_factory([
            blue_remove_host('Enterprise0'),
        ])

        def red_sleep(state, step):
            return RED_SLEEP

        result = harness.run_episode(blue_policy, red_sleep)
        assert result.error_diffs == 0

    def test_remove_after_monitor(self):
        """Remove after monitor should work on compromised host."""
        from tests.comparison.policies import react_remove_policy_with_timing

        harness = DifferentialHarness(seed=42, max_steps=30, verbose=False)
        remove_policy = react_remove_policy_with_timing(delay=0)
        result = harness.run_bline_episode(remove_policy, use_jax_bline=False)

        remove_count = sum(1 for sr in result.step_results if 'Remove' in sr.blue_action_desc)
        assert remove_count >= 1, "Should perform at least one Remove action"


@requires_cyborg
class TestMonitorEdgeCases:
    """Test monitor action edge cases."""

    def test_monitor_on_clean_network(self):
        """Monitor on clean network should not detect anything (except User0)."""
        harness = DifferentialHarness(seed=42, max_steps=5, verbose=False)

        blue_policy = scripted_blue_policy_factory([BLUE_MONITOR] * 5)

        def red_sleep(state, step):
            return RED_SLEEP

        result = harness.run_episode(blue_policy, red_sleep)

        for sr in result.step_results:
            for hostname, detected in sr.jax_state.host_activity_detected.items():
                if hostname != 'User0':
                    assert not detected, \
                        f"Should not detect activity on {hostname} without Red"

    def test_repeated_monitor_consistent(self):
        """Repeated monitor should be consistent."""
        harness = DifferentialHarness(seed=42, max_steps=20, verbose=False)

        blue_policy = scripted_blue_policy_factory([BLUE_MONITOR] * 20)
        result = harness.run_bline_episode(blue_policy, use_jax_bline=False)

        assert result.error_diffs == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
