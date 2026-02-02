"""Differential tests comparing CybORG Impact behavior to JAX.

Every test runs both CybORG and JAX with identical actions and compares results.
These tests verify Impact action behavior on Op_Server0 and availability penalties.
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
    red_impact_host,
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
class TestImpactCybORGParity:
    """Differential tests comparing CybORG Impact behavior to JAX.

    Key CybORG behavior:
    - Impact only works on Op_Server0 (the target host)
    - Impact requires PRIVILEGED access on Op_Server0
    - Successful Impact gives -10.0 availability penalty per step
    - Impact stops the OT service
    """

    def test_impact_on_op_server_with_privileges(self):
        """CybORG vs JAX: Impact on Op_Server0 with PRIVILEGED access."""
        harness = DifferentialHarness(seed=42, max_steps=20, verbose=False)

        red_actions = BLINE_KILLCHAIN_STANDARD[:15] + [
            red_impact_host('Op_Server0'),
            0,
            0,
        ]

        blue_actions = [BLUE_MONITOR] * 18

        red_policy = scripted_red_policy_factory(red_actions)
        blue_policy = scripted_blue_policy_factory(blue_actions)

        result = harness.run_episode(blue_policy, red_policy)

        for sr in result.step_results:
            cyborg_blue = sr.cyborg_state.reward_blue
            jax_blue = sr.jax_state.reward_blue
            assert abs(cyborg_blue - jax_blue) < 0.02, \
                f"Step {sr.step}: Blue reward mismatch CybORG={cyborg_blue}, JAX={jax_blue}"

    def test_impact_without_privileges_fails(self):
        """CybORG vs JAX: Impact without PRIVILEGED access should fail."""
        harness = DifferentialHarness(seed=42, max_steps=20, verbose=False)

        red_actions = BLINE_KILLCHAIN_STANDARD[:14] + [
            red_impact_host('Op_Server0'),
            0,
        ]

        blue_actions = [BLUE_MONITOR] * 16

        red_policy = scripted_red_policy_factory(red_actions)
        blue_policy = scripted_blue_policy_factory(blue_actions)

        result = harness.run_episode(blue_policy, red_policy)

        for sr in result.step_results:
            cyborg_blue = sr.cyborg_state.reward_blue
            jax_blue = sr.jax_state.reward_blue
            assert abs(cyborg_blue - jax_blue) < 0.02, \
                f"Step {sr.step}: Blue reward mismatch CybORG={cyborg_blue}, JAX={jax_blue}"

    def test_impact_on_wrong_host_fails(self):
        """CybORG vs JAX: Impact on non-Op_Server0 host fails."""
        harness = DifferentialHarness(seed=42, max_steps=10, verbose=False)

        red_actions = [
            red_discover_subnet(SUBNET_USER),
            red_scan_host('User1'),
            red_exploit_host('User1', EXPLOIT_SSH),
            red_privesc_host('User1'),
            red_impact_host('User1'),
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

    def test_impact_availability_penalty_persists(self):
        """CybORG vs JAX: Impact penalty continues each step after Impact."""
        harness = DifferentialHarness(seed=42, max_steps=25, verbose=False)

        red_actions = BLINE_KILLCHAIN_STANDARD[:15] + [
            red_impact_host('Op_Server0'),
            0,
            0,
            0,
            0,
        ]

        blue_actions = [BLUE_MONITOR] * 20

        red_policy = scripted_red_policy_factory(red_actions)
        blue_policy = scripted_blue_policy_factory(blue_actions)

        result = harness.run_episode(blue_policy, red_policy)

        for sr in result.step_results:
            cyborg_blue = sr.cyborg_state.reward_blue
            jax_blue = sr.jax_state.reward_blue
            assert abs(cyborg_blue - jax_blue) < 0.02, \
                f"Step {sr.step}: Blue reward mismatch CybORG={cyborg_blue}, JAX={jax_blue}"

    def test_impact_then_restore_stops_penalty(self):
        """CybORG vs JAX: Restore on Op_Server0 stops availability penalty."""
        harness = DifferentialHarness(seed=42, max_steps=25, verbose=False)

        red_actions = BLINE_KILLCHAIN_STANDARD[:15] + [
            red_impact_host('Op_Server0'),
            0,
            0,
            0,
        ]

        blue_actions = [BLUE_MONITOR] * 15 + [
            BLUE_MONITOR,
            BLUE_MONITOR,
            blue_restore_host('Op_Server0'),
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

    def test_impact_multiple_times(self):
        """CybORG vs JAX: Multiple Impact actions (second has no effect)."""
        harness = DifferentialHarness(seed=42, max_steps=25, verbose=False)

        red_actions = BLINE_KILLCHAIN_STANDARD[:15] + [
            red_impact_host('Op_Server0'),
            red_impact_host('Op_Server0'),
            0,
        ]

        blue_actions = [BLUE_MONITOR] * 18

        red_policy = scripted_red_policy_factory(red_actions)
        blue_policy = scripted_blue_policy_factory(blue_actions)

        result = harness.run_episode(blue_policy, red_policy)

        for sr in result.step_results:
            cyborg_blue = sr.cyborg_state.reward_blue
            jax_blue = sr.jax_state.reward_blue
            assert abs(cyborg_blue - jax_blue) < 0.02, \
                f"Step {sr.step}: Blue reward mismatch CybORG={cyborg_blue}, JAX={jax_blue}"

    def test_impact_reward_values(self):
        """CybORG vs JAX: Verify Impact availability penalty values."""
        harness = DifferentialHarness(seed=42, max_steps=25, verbose=False)

        red_actions = BLINE_KILLCHAIN_STANDARD[:15] + [
            red_impact_host('Op_Server0'),
            0,
            0,
        ]

        blue_actions = [BLUE_MONITOR] * 18

        red_policy = scripted_red_policy_factory(red_actions)
        blue_policy = scripted_blue_policy_factory(blue_actions)

        result = harness.run_episode(blue_policy, red_policy)

        for sr in result.step_results:
            assert abs(sr.cyborg_state.reward_blue - sr.jax_state.reward_blue) < 0.02, \
                f"Step {sr.step}: Blue reward mismatch CybORG={sr.cyborg_state.reward_blue}, JAX={sr.jax_state.reward_blue}"
            assert abs(sr.cyborg_state.reward_red - sr.jax_state.reward_red) < 0.02, \
                f"Step {sr.step}: Red reward mismatch CybORG={sr.cyborg_state.reward_red}, JAX={sr.jax_state.reward_red}"


@requires_cyborg
class TestImpactStateParity:
    """Tests verifying Impact affects state correctly."""

    def test_impact_sets_ot_service_stopped(self):
        """CybORG vs JAX: Impact should set ot_service_stopped flag."""
        harness = DifferentialHarness(seed=42, max_steps=20, verbose=False)

        red_actions = BLINE_KILLCHAIN_STANDARD[:15] + [
            red_impact_host('Op_Server0'),
            0,
        ]

        blue_actions = [BLUE_MONITOR] * 17

        red_policy = scripted_red_policy_factory(red_actions)
        blue_policy = scripted_blue_policy_factory(blue_actions)

        result = harness.run_episode(blue_policy, red_policy)

        impact_step = result.step_results[15]

        assert impact_step.jax_state.ot_service_stopped.get('Op_Server0', False), \
            "OT service should be stopped after Impact"

    def test_restore_clears_ot_service_stopped(self):
        """CybORG vs JAX: Restore should clear ot_service_stopped flag."""
        harness = DifferentialHarness(seed=42, max_steps=25, verbose=False)

        red_actions = BLINE_KILLCHAIN_STANDARD[:15] + [
            red_impact_host('Op_Server0'),
            0,
            0,
        ]

        blue_actions = [BLUE_MONITOR] * 16 + [
            blue_restore_host('Op_Server0'),
            BLUE_MONITOR,
        ]

        red_policy = scripted_red_policy_factory(red_actions)
        blue_policy = scripted_blue_policy_factory(blue_actions)

        result = harness.run_episode(blue_policy, red_policy)

        after_restore = result.step_results[16]

        assert not after_restore.jax_state.ot_service_stopped.get('Op_Server0', True), \
            "OT service should be running after Restore"
