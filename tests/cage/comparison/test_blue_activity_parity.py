"""Tests verifying host_activity_detected parity between CybORG and JAX.

When Red exploits a host and Blue monitors, both environments should
agree on which hosts have detected activity.
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
    blue_remove_host,
    blue_restore_host,
    SUBNET_USER,
    EXPLOIT_SSH,
    BLINE_KILLCHAIN_STANDARD,
)
from jaxmarl.environments.cage.actions import BLUE_MONITOR


requires_cyborg = pytest.mark.skipif(
    not is_cyborg_available(),
    reason="CybORG not installed"
)


@requires_cyborg
class TestExploitActivityDetection:
    """Blue should detect exploit activity when monitoring."""

    def test_exploit_detected_by_monitor(self):
        """After Red exploits User1 and Blue monitors, both environments
        should agree on host_activity_detected for User1."""
        harness = DifferentialHarness(
            seed=42, max_steps=8, verbose=False, sync_detection_rng=True,
        )

        red_actions = [
            red_discover_subnet(SUBNET_USER),
            red_scan_host('User1'),
            red_exploit_host('User1', EXPLOIT_SSH),
            0,
        ]
        blue_actions = [BLUE_MONITOR] * len(red_actions)

        result = harness.run_episode(
            scripted_blue_policy_factory(blue_actions),
            scripted_red_policy_factory(red_actions),
        )

        step3 = result.step_results[2]
        cyborg_det = step3.cyborg_state.host_activity_detected['User1']
        jax_det = step3.jax_state.host_activity_detected['User1']
        assert cyborg_det == jax_det, (
            f"host_activity_detected mismatch for User1 at step 3: "
            f"CybORG={cyborg_det}, JAX={jax_det}"
        )

    def test_activity_persists_across_steps(self):
        """Once exploit activity is detected, it should persist in both
        environments until cleared by Remove/Restore."""
        harness = DifferentialHarness(
            seed=42, max_steps=10, verbose=False, sync_detection_rng=True,
        )

        red_actions = [
            red_discover_subnet(SUBNET_USER),
            red_scan_host('User1'),
            red_exploit_host('User1', EXPLOIT_SSH),
            0,
            0,
        ]
        blue_actions = [BLUE_MONITOR] * len(red_actions)

        result = harness.run_episode(
            scripted_blue_policy_factory(blue_actions),
            scripted_red_policy_factory(red_actions),
        )

        for sr in result.step_results[2:]:
            cyborg_det = sr.cyborg_state.host_activity_detected['User1']
            jax_det = sr.jax_state.host_activity_detected['User1']
            assert cyborg_det == jax_det, (
                f"host_activity_detected mismatch at step {sr.step}: "
                f"CybORG={cyborg_det}, JAX={jax_det}"
            )

    def test_no_activity_without_exploit(self):
        """Scan-only (no exploit) should not trigger host_activity_detected."""
        harness = DifferentialHarness(
            seed=42, max_steps=5, verbose=False, sync_detection_rng=True,
        )

        red_actions = [
            red_discover_subnet(SUBNET_USER),
            red_scan_host('User1'),
            0,
        ]
        blue_actions = [BLUE_MONITOR] * len(red_actions)

        result = harness.run_episode(
            scripted_blue_policy_factory(blue_actions),
            scripted_red_policy_factory(red_actions),
        )

        for sr in result.step_results:
            activity_diffs = [
                d for d in sr.diffs
                if d.field == 'host_activity_detected'
            ]
            assert len(activity_diffs) == 0, (
                f"Step {sr.step}: unexpected host_activity_detected diff: "
                + "; ".join(str(d) for d in activity_diffs)
            )


@requires_cyborg
class TestActivityClearedByDefense:
    """Remove and Restore should clear host_activity_detected."""

    def test_activity_cleared_by_remove(self):
        """After Remove, host_activity_detected should be cleared in both."""
        harness = DifferentialHarness(
            seed=42, max_steps=10, verbose=False, sync_detection_rng=True,
        )

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
            blue_remove_host('User1'),
            BLUE_MONITOR,
        ]

        result = harness.run_episode(
            scripted_blue_policy_factory(blue_actions),
            scripted_red_policy_factory(red_actions),
        )

        step5 = result.step_results[4]
        cyborg_det = step5.cyborg_state.host_activity_detected['User1']
        jax_det = step5.jax_state.host_activity_detected['User1']
        assert cyborg_det == jax_det, (
            f"After Remove, host_activity_detected mismatch: "
            f"CybORG={cyborg_det}, JAX={jax_det}"
        )

    def test_activity_cleared_by_restore(self):
        """After Restore, host_activity_detected should be cleared in both."""
        harness = DifferentialHarness(
            seed=42, max_steps=10, verbose=False, sync_detection_rng=True,
        )

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
            blue_restore_host('User1'),
            BLUE_MONITOR,
        ]

        result = harness.run_episode(
            scripted_blue_policy_factory(blue_actions),
            scripted_red_policy_factory(red_actions),
        )

        step5 = result.step_results[4]
        cyborg_det = step5.cyborg_state.host_activity_detected['User1']
        jax_det = step5.jax_state.host_activity_detected['User1']
        assert cyborg_det == jax_det, (
            f"After Restore, host_activity_detected mismatch: "
            f"CybORG={cyborg_det}, JAX={jax_det}"
        )


@requires_cyborg
class TestBlineKillchainActivity:
    """B_line killchain should not produce host_activity_detected warnings."""

    def test_no_activity_warnings_in_bline_killchain(self):
        """Run B_line killchain with Monitor and verify no activity mismatches."""
        harness = DifferentialHarness(
            seed=42, max_steps=20, verbose=False, sync_detection_rng=True,
        )

        red_actions = BLINE_KILLCHAIN_STANDARD[:15]
        blue_actions = [BLUE_MONITOR] * len(red_actions)

        result = harness.run_episode(
            scripted_blue_policy_factory(blue_actions),
            scripted_red_policy_factory(red_actions),
        )

        activity_warnings = [
            d for sr in result.step_results for d in sr.diffs
            if d.field == 'host_activity_detected'
        ]
        assert len(activity_warnings) == 0, (
            f"Expected no host_activity_detected warnings, got {len(activity_warnings)}: "
            + "; ".join(str(d) for d in activity_warnings)
        )
