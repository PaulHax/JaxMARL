"""Differential smoke tests for B_line and Meander agent trajectories.

These tests run complete episodes with CybORG's scripted agents and verify
that JAX produces matching rewards and state transitions.
"""

import pytest
from tests.differential.harness import DifferentialHarness, is_cyborg_available
from tests.comparison.policies import (
    scripted_blue_policy_factory,
    scripted_red_policy_factory,
)
from tests.comparison.scenarios import (
    BLINE_KILLCHAIN_STANDARD,
    KILLCHAIN_VIA_ENTERPRISE0,
    KILLCHAIN_VIA_HARAKA,
    blue_remove_host,
    blue_restore_host,
)
from jaxmarl.environments.cage.actions import BLUE_SLEEP, BLUE_MONITOR


requires_cyborg = pytest.mark.skipif(
    not is_cyborg_available(),
    reason="CybORG not installed"
)


@requires_cyborg
class TestBLineSmokeParity:
    """Smoke tests comparing B_line agent behavior between CybORG and JAX."""

    def test_bline_standard_killchain_passive_blue(self):
        """CybORG vs JAX: B_line killchain with passive Blue."""
        harness = DifferentialHarness(seed=42, max_steps=20, verbose=False)

        red_actions = BLINE_KILLCHAIN_STANDARD[:16]
        blue_actions = [BLUE_SLEEP] * 16

        red_policy = scripted_red_policy_factory(red_actions)
        blue_policy = scripted_blue_policy_factory(blue_actions)

        result = harness.run_episode(blue_policy, red_policy)

        for sr in result.step_results:
            cyborg_blue = sr.cyborg_state.reward_blue
            jax_blue = sr.jax_state.reward_blue
            assert abs(cyborg_blue - jax_blue) < 0.02, \
                f"Step {sr.step}: Blue reward mismatch CybORG={cyborg_blue}, JAX={jax_blue}"

    def test_bline_standard_killchain_monitor_blue(self):
        """CybORG vs JAX: B_line killchain with monitoring Blue."""
        harness = DifferentialHarness(seed=42, max_steps=20, verbose=False)

        red_actions = BLINE_KILLCHAIN_STANDARD[:16]
        blue_actions = [BLUE_MONITOR] * 16

        red_policy = scripted_red_policy_factory(red_actions)
        blue_policy = scripted_blue_policy_factory(blue_actions)

        result = harness.run_episode(blue_policy, red_policy)

        for sr in result.step_results:
            cyborg_blue = sr.cyborg_state.reward_blue
            jax_blue = sr.jax_state.reward_blue
            assert abs(cyborg_blue - jax_blue) < 0.02, \
                f"Step {sr.step}: Blue reward mismatch CybORG={cyborg_blue}, JAX={jax_blue}"

    def test_bline_enterprise0_path(self):
        """CybORG vs JAX: B_line via Enterprise0 path."""
        harness = DifferentialHarness(seed=42, max_steps=20, verbose=False)

        red_actions = KILLCHAIN_VIA_ENTERPRISE0[:14]
        blue_actions = [BLUE_MONITOR] * 14

        red_policy = scripted_red_policy_factory(red_actions)
        blue_policy = scripted_blue_policy_factory(blue_actions)

        result = harness.run_episode(blue_policy, red_policy)

        for sr in result.step_results:
            cyborg_blue = sr.cyborg_state.reward_blue
            jax_blue = sr.jax_state.reward_blue
            assert abs(cyborg_blue - jax_blue) < 0.02, \
                f"Step {sr.step}: Blue reward mismatch CybORG={cyborg_blue}, JAX={jax_blue}"

    def test_bline_haraka_path(self):
        """CybORG vs JAX: B_line via Haraka exploit path."""
        harness = DifferentialHarness(seed=42, max_steps=20, verbose=False)

        red_actions = KILLCHAIN_VIA_HARAKA[:14]
        blue_actions = [BLUE_MONITOR] * 14

        red_policy = scripted_red_policy_factory(red_actions)
        blue_policy = scripted_blue_policy_factory(blue_actions)

        result = harness.run_episode(blue_policy, red_policy)

        for sr in result.step_results:
            cyborg_blue = sr.cyborg_state.reward_blue
            jax_blue = sr.jax_state.reward_blue
            assert abs(cyborg_blue - jax_blue) < 0.02, \
                f"Step {sr.step}: Blue reward mismatch CybORG={cyborg_blue}, JAX={jax_blue}"


@requires_cyborg
class TestBLineWithDefenseParity:
    """Smoke tests for B_line killchain with Blue defense actions."""

    def test_bline_with_early_remove(self):
        """CybORG vs JAX: B_line killchain with early Remove intervention."""
        harness = DifferentialHarness(seed=42, max_steps=20, verbose=False)

        red_actions = BLINE_KILLCHAIN_STANDARD[:16]
        blue_actions = [
            BLUE_MONITOR,
            BLUE_MONITOR,
            BLUE_MONITOR,
            blue_remove_host('User1'),
        ] + [BLUE_MONITOR] * 12

        red_policy = scripted_red_policy_factory(red_actions)
        blue_policy = scripted_blue_policy_factory(blue_actions)

        result = harness.run_episode(blue_policy, red_policy)

        for sr in result.step_results:
            cyborg_blue = sr.cyborg_state.reward_blue
            jax_blue = sr.jax_state.reward_blue
            assert abs(cyborg_blue - jax_blue) < 0.02, \
                f"Step {sr.step}: Blue reward mismatch CybORG={cyborg_blue}, JAX={jax_blue}"

    def test_bline_with_early_restore(self):
        """CybORG vs JAX: B_line killchain with early Restore intervention."""
        harness = DifferentialHarness(seed=42, max_steps=20, verbose=False)

        red_actions = BLINE_KILLCHAIN_STANDARD[:16]
        blue_actions = [
            BLUE_MONITOR,
            BLUE_MONITOR,
            BLUE_MONITOR,
            BLUE_MONITOR,
            blue_restore_host('User1'),
        ] + [BLUE_MONITOR] * 11

        red_policy = scripted_red_policy_factory(red_actions)
        blue_policy = scripted_blue_policy_factory(blue_actions)

        result = harness.run_episode(blue_policy, red_policy)

        for sr in result.step_results:
            cyborg_blue = sr.cyborg_state.reward_blue
            jax_blue = sr.jax_state.reward_blue
            assert abs(cyborg_blue - jax_blue) < 0.02, \
                f"Step {sr.step}: Blue reward mismatch CybORG={cyborg_blue}, JAX={jax_blue}"

    def test_bline_with_multiple_restores(self):
        """CybORG vs JAX: B_line with multiple Restore actions."""
        harness = DifferentialHarness(seed=42, max_steps=20, verbose=False)

        red_actions = BLINE_KILLCHAIN_STANDARD[:16]
        blue_actions = [
            BLUE_MONITOR,
            BLUE_MONITOR,
            BLUE_MONITOR,
            BLUE_MONITOR,
            blue_restore_host('User1'),
            BLUE_MONITOR,
            BLUE_MONITOR,
            BLUE_MONITOR,
            blue_restore_host('Enterprise1'),
            BLUE_MONITOR,
            BLUE_MONITOR,
            BLUE_MONITOR,
            blue_restore_host('Enterprise2'),
            BLUE_MONITOR,
            BLUE_MONITOR,
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


@requires_cyborg
class TestLongEpisodeParity:
    """Smoke tests for longer episodes to verify trajectory matching."""

    def test_50_step_episode(self):
        """CybORG vs JAX: 50 step episode with B_line."""
        harness = DifferentialHarness(seed=42, max_steps=50, verbose=False)

        red_actions = BLINE_KILLCHAIN_STANDARD[:16] + [0] * 34
        blue_actions = [BLUE_MONITOR] * 50

        red_policy = scripted_red_policy_factory(red_actions)
        blue_policy = scripted_blue_policy_factory(blue_actions)

        result = harness.run_episode(blue_policy, red_policy)

        for sr in result.step_results:
            cyborg_blue = sr.cyborg_state.reward_blue
            jax_blue = sr.jax_state.reward_blue
            assert abs(cyborg_blue - jax_blue) < 0.02, \
                f"Step {sr.step}: Blue reward mismatch CybORG={cyborg_blue}, JAX={jax_blue}"

    def test_100_step_episode_with_defense(self):
        """CybORG vs JAX: 100 step episode with varied actions."""
        harness = DifferentialHarness(seed=42, max_steps=100, verbose=False)

        red_actions = BLINE_KILLCHAIN_STANDARD[:16] + [0] * 84

        blue_actions = [BLUE_MONITOR] * 20 + [
            blue_restore_host('User1'),
        ] + [BLUE_MONITOR] * 20 + [
            blue_restore_host('Enterprise1'),
        ] + [BLUE_MONITOR] * 20 + [
            blue_restore_host('Enterprise2'),
        ] + [BLUE_MONITOR] * 37

        red_policy = scripted_red_policy_factory(red_actions)
        blue_policy = scripted_blue_policy_factory(blue_actions)

        result = harness.run_episode(blue_policy, red_policy)

        for sr in result.step_results:
            cyborg_blue = sr.cyborg_state.reward_blue
            jax_blue = sr.jax_state.reward_blue
            assert abs(cyborg_blue - jax_blue) < 0.02, \
                f"Step {sr.step}: Blue reward mismatch CybORG={cyborg_blue}, JAX={jax_blue}"


@requires_cyborg
class TestTotalRewardParity:
    """Tests comparing total episode rewards."""

    def test_total_reward_passive_defense(self):
        """CybORG vs JAX: Total rewards with passive Blue."""
        harness = DifferentialHarness(seed=42, max_steps=20, verbose=False)

        red_actions = BLINE_KILLCHAIN_STANDARD[:16]
        blue_actions = [BLUE_SLEEP] * 16

        red_policy = scripted_red_policy_factory(red_actions)
        blue_policy = scripted_blue_policy_factory(blue_actions)

        result = harness.run_episode(blue_policy, red_policy)

        cyborg_blue_total = sum(sr.cyborg_state.reward_blue for sr in result.step_results)
        jax_blue_total = sum(sr.jax_state.reward_blue for sr in result.step_results)

        assert abs(cyborg_blue_total - jax_blue_total) < 1.0, \
            f"Total Blue reward mismatch: CybORG={cyborg_blue_total:.2f}, JAX={jax_blue_total:.2f}"

    def test_total_reward_active_defense(self):
        """CybORG vs JAX: Total rewards with active Blue defense."""
        harness = DifferentialHarness(seed=42, max_steps=20, verbose=False)

        red_actions = BLINE_KILLCHAIN_STANDARD[:16]
        blue_actions = [
            BLUE_MONITOR,
            BLUE_MONITOR,
            BLUE_MONITOR,
            BLUE_MONITOR,
            blue_restore_host('User1'),
            BLUE_MONITOR,
            BLUE_MONITOR,
            BLUE_MONITOR,
            blue_restore_host('Enterprise1'),
        ] + [BLUE_MONITOR] * 7

        red_policy = scripted_red_policy_factory(red_actions)
        blue_policy = scripted_blue_policy_factory(blue_actions)

        result = harness.run_episode(blue_policy, red_policy)

        cyborg_blue_total = sum(sr.cyborg_state.reward_blue for sr in result.step_results)
        jax_blue_total = sum(sr.jax_state.reward_blue for sr in result.step_results)

        assert abs(cyborg_blue_total - jax_blue_total) < 1.0, \
            f"Total Blue reward mismatch: CybORG={cyborg_blue_total:.2f}, JAX={jax_blue_total:.2f}"


@requires_cyborg
class TestMultipleSeedsParity:
    """Tests running same scenario with different seeds."""

    @pytest.mark.parametrize("seed", [1, 42, 123, 456, 789])
    def test_bline_killchain_different_seeds(self, seed):
        """CybORG vs JAX: B_line killchain with different seeds."""
        harness = DifferentialHarness(seed=seed, max_steps=20, verbose=False)

        red_actions = BLINE_KILLCHAIN_STANDARD[:16]
        blue_actions = [BLUE_MONITOR] * 16

        red_policy = scripted_red_policy_factory(red_actions)
        blue_policy = scripted_blue_policy_factory(blue_actions)

        result = harness.run_episode(blue_policy, red_policy)

        for sr in result.step_results:
            cyborg_blue = sr.cyborg_state.reward_blue
            jax_blue = sr.jax_state.reward_blue
            assert abs(cyborg_blue - jax_blue) < 0.02, \
                f"Seed {seed}, Step {sr.step}: Blue reward mismatch CybORG={cyborg_blue}, JAX={jax_blue}"
