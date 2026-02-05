"""Differential tests comparing CybORG observation encoding to JAX.

Every test runs both CybORG and JAX with identical actions and compares
Blue agent observations after various scenarios.
"""

import pytest
from tests.cage.differential.harness import (
    DifferentialHarness,
    is_cyborg_available,
    sleep_policy,
)
from tests.cage.comparison.policies import (
    scripted_blue_policy_factory,
    scripted_red_policy_factory,
)
from tests.cage.comparison.scenarios import (
    red_discover_subnet,
    red_scan_host,
    red_exploit_host,
    red_privesc_host,
    blue_analyse_host,
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
class TestObservationParity:
    """Differential tests for Blue observation encoding."""

    def test_initial_observation(self):
        """CybORG vs JAX: Initial observation should match."""
        harness = DifferentialHarness(seed=42, max_steps=5, check_obs=True, verbose=False)

        red_actions = [0]
        blue_actions = [BLUE_MONITOR]

        red_policy = scripted_red_policy_factory(red_actions)
        blue_policy = scripted_blue_policy_factory(blue_actions)

        result = harness.run_episode(blue_policy, red_policy)

        initial_step = result.step_results[0]

        assert abs(initial_step.cyborg_state.reward_blue - initial_step.jax_state.reward_blue) < 0.02, \
            f"Blue reward mismatch: CybORG={initial_step.cyborg_state.reward_blue}, JAX={initial_step.jax_state.reward_blue}"

    @pytest.mark.xfail(reason="Known mismatch at blue_obs index 22 (Op_Host1 compromised_0)")
    def test_blue_obs_index_22_parity(self):
        """Targeted parity check for Blue obs index 22 (Op_Host1 compromised_0)."""
        harness = DifferentialHarness(seed=42, max_steps=1, check_obs=True, verbose=False)

        result = harness.run_bline_episode(sleep_policy, use_jax_bline=False)
        step = result.step_results[0]

        assert step.cyborg_state.blue_obs is not None
        assert step.jax_state.blue_obs is not None

        cyborg_val = float(step.cyborg_state.blue_obs[22])
        jax_val = float(step.jax_state.blue_obs[22])

        assert abs(cyborg_val - jax_val) < 1e-6, (
            f"blue_obs[22] mismatch: CybORG={cyborg_val}, JAX={jax_val}"
        )

    def test_observation_after_exploit(self):
        """CybORG vs JAX: Observation after Red exploit."""
        harness = DifferentialHarness(seed=42, max_steps=10, check_obs=True, verbose=False)

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
            assert abs(sr.cyborg_state.reward_blue - sr.jax_state.reward_blue) < 0.02, \
                f"Step {sr.step}: Blue reward mismatch CybORG={sr.cyborg_state.reward_blue}, JAX={sr.jax_state.reward_blue}"

    def test_observation_after_analyse(self):
        """CybORG vs JAX: Observation after Blue Analyse detects compromise."""
        harness = DifferentialHarness(seed=42, max_steps=10, check_obs=True, verbose=False)

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
            blue_analyse_host('User1'),
            BLUE_MONITOR,
        ]

        red_policy = scripted_red_policy_factory(red_actions)
        blue_policy = scripted_blue_policy_factory(blue_actions)

        result = harness.run_episode(blue_policy, red_policy)

        for sr in result.step_results:
            assert abs(sr.cyborg_state.reward_blue - sr.jax_state.reward_blue) < 0.02, \
                f"Step {sr.step}: Blue reward mismatch CybORG={sr.cyborg_state.reward_blue}, JAX={sr.jax_state.reward_blue}"

    def test_observation_after_remove(self):
        """CybORG vs JAX: Observation after Blue Remove clears compromise."""
        harness = DifferentialHarness(seed=42, max_steps=10, check_obs=True, verbose=False)

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

        red_policy = scripted_red_policy_factory(red_actions)
        blue_policy = scripted_blue_policy_factory(blue_actions)

        result = harness.run_episode(blue_policy, red_policy)

        for sr in result.step_results:
            assert abs(sr.cyborg_state.reward_blue - sr.jax_state.reward_blue) < 0.02, \
                f"Step {sr.step}: Blue reward mismatch CybORG={sr.cyborg_state.reward_blue}, JAX={sr.jax_state.reward_blue}"

    def test_observation_after_restore(self):
        """CybORG vs JAX: Observation after Blue Restore resets host."""
        harness = DifferentialHarness(seed=42, max_steps=10, check_obs=True, verbose=False)

        red_actions = [
            red_discover_subnet(SUBNET_USER),
            red_scan_host('User1'),
            red_exploit_host('User1', EXPLOIT_SSH),
            red_privesc_host('User1'),
            0,
        ]

        blue_actions = [
            BLUE_MONITOR,
            BLUE_MONITOR,
            BLUE_MONITOR,
            BLUE_MONITOR,
            blue_restore_host('User1'),
        ]

        red_policy = scripted_red_policy_factory(red_actions)
        blue_policy = scripted_blue_policy_factory(blue_actions)

        result = harness.run_episode(blue_policy, red_policy)

        for sr in result.step_results:
            assert abs(sr.cyborg_state.reward_blue - sr.jax_state.reward_blue) < 0.02, \
                f"Step {sr.step}: Blue reward mismatch CybORG={sr.cyborg_state.reward_blue}, JAX={sr.jax_state.reward_blue}"

    def test_observation_multiple_hosts_compromised(self):
        """CybORG vs JAX: Observation with multiple compromised hosts."""
        harness = DifferentialHarness(seed=42, max_steps=15, check_obs=True, verbose=False)

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
            assert abs(sr.cyborg_state.reward_blue - sr.jax_state.reward_blue) < 0.02, \
                f"Step {sr.step}: Blue reward mismatch CybORG={sr.cyborg_state.reward_blue}, JAX={sr.jax_state.reward_blue}"

    def test_observation_privileged_vs_user(self):
        """CybORG vs JAX: Observation distinguishes privileged from user access."""
        harness = DifferentialHarness(seed=42, max_steps=12, check_obs=True, verbose=False)

        red_actions = [
            red_discover_subnet(SUBNET_USER),
            red_scan_host('User1'),
            red_exploit_host('User1', EXPLOIT_SSH),
            red_privesc_host('User1'),
            0,
            0,
        ]

        blue_actions = [
            BLUE_MONITOR,
            BLUE_MONITOR,
            BLUE_MONITOR,
            BLUE_MONITOR,
            blue_analyse_host('User1'),
            BLUE_MONITOR,
        ]

        red_policy = scripted_red_policy_factory(red_actions)
        blue_policy = scripted_blue_policy_factory(blue_actions)

        result = harness.run_episode(blue_policy, red_policy)

        for sr in result.step_results:
            assert abs(sr.cyborg_state.reward_blue - sr.jax_state.reward_blue) < 0.02, \
                f"Step {sr.step}: Blue reward mismatch CybORG={sr.cyborg_state.reward_blue}, JAX={sr.jax_state.reward_blue}"


@requires_cyborg
class TestActivityDetectionParity:
    """Tests for activity detection in observations."""

    def test_activity_detected_on_scan(self):
        """CybORG vs JAX: Activity detection on host scan."""
        harness = DifferentialHarness(seed=42, max_steps=8, verbose=False)

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
            assert abs(sr.cyborg_state.reward_blue - sr.jax_state.reward_blue) < 0.02, \
                f"Step {sr.step}: Blue reward mismatch CybORG={sr.cyborg_state.reward_blue}, JAX={sr.jax_state.reward_blue}"

    def test_activity_detected_on_exploit(self):
        """CybORG vs JAX: Activity detection on exploit attempt."""
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
            assert abs(sr.cyborg_state.reward_blue - sr.jax_state.reward_blue) < 0.02, \
                f"Step {sr.step}: Blue reward mismatch CybORG={sr.cyborg_state.reward_blue}, JAX={sr.jax_state.reward_blue}"

    def test_activity_clears_after_restore(self):
        """CybORG vs JAX: Activity detection clears after Restore."""
        harness = DifferentialHarness(seed=42, max_steps=10, verbose=False)

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

        red_policy = scripted_red_policy_factory(red_actions)
        blue_policy = scripted_blue_policy_factory(blue_actions)

        result = harness.run_episode(blue_policy, red_policy)

        for sr in result.step_results:
            assert abs(sr.cyborg_state.reward_blue - sr.jax_state.reward_blue) < 0.02, \
                f"Step {sr.step}: Blue reward mismatch CybORG={sr.cyborg_state.reward_blue}, JAX={sr.jax_state.reward_blue}"


@requires_cyborg
class TestObservationThroughKillchainParity:
    """Tests observation encoding through complete killchain scenarios."""

    def test_observation_through_bline_killchain(self):
        """CybORG vs JAX: Observation encoding through B_line killchain."""
        harness = DifferentialHarness(seed=42, max_steps=20, check_obs=True, verbose=False)

        red_actions = BLINE_KILLCHAIN_STANDARD[:16]
        blue_actions = [BLUE_MONITOR] * 16

        red_policy = scripted_red_policy_factory(red_actions)
        blue_policy = scripted_blue_policy_factory(blue_actions)

        result = harness.run_episode(blue_policy, red_policy)

        for sr in result.step_results:
            assert abs(sr.cyborg_state.reward_blue - sr.jax_state.reward_blue) < 0.02, \
                f"Step {sr.step}: Blue reward mismatch CybORG={sr.cyborg_state.reward_blue}, JAX={sr.jax_state.reward_blue}"

    def test_observation_with_mixed_blue_actions(self):
        """CybORG vs JAX: Observation with varied Blue responses."""
        harness = DifferentialHarness(seed=42, max_steps=15, check_obs=True, verbose=False)

        red_actions = [
            red_discover_subnet(SUBNET_USER),
            red_scan_host('User1'),
            red_exploit_host('User1', EXPLOIT_SSH),
            red_privesc_host('User1'),
            red_scan_host('User2'),
            red_exploit_host('User2', EXPLOIT_SSH),
            0,
            0,
        ]

        blue_actions = [
            BLUE_MONITOR,
            BLUE_MONITOR,
            BLUE_MONITOR,
            blue_analyse_host('User1'),
            blue_remove_host('User1'),
            BLUE_MONITOR,
            blue_analyse_host('User2'),
            blue_restore_host('User2'),
        ]

        red_policy = scripted_red_policy_factory(red_actions)
        blue_policy = scripted_blue_policy_factory(blue_actions)

        result = harness.run_episode(blue_policy, red_policy)

        for sr in result.step_results:
            assert abs(sr.cyborg_state.reward_blue - sr.jax_state.reward_blue) < 0.02, \
                f"Step {sr.step}: Blue reward mismatch CybORG={sr.cyborg_state.reward_blue}, JAX={sr.jax_state.reward_blue}"
