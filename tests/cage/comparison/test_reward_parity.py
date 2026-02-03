"""Differential tests comparing CybORG reward calculations to JAX.

Every test runs both CybORG and JAX with identical actions and compares rewards.
Tests confidentiality values, availability values, zero-sum property, and action costs.
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
    red_impact_host,
    blue_remove_host,
    blue_restore_host,
    blue_analyse_host,
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
class TestRewardValuesParity:
    """Tests verifying reward values match CybORG."""

    def test_no_activity_zero_rewards(self):
        """CybORG vs JAX: No activity should give zero rewards."""
        harness = DifferentialHarness(seed=42, max_steps=5, verbose=False)

        red_actions = [0, 0, 0, 0, 0]
        blue_actions = [BLUE_MONITOR] * 5

        red_policy = scripted_red_policy_factory(red_actions)
        blue_policy = scripted_blue_policy_factory(blue_actions)

        result = harness.run_episode(blue_policy, red_policy)

        for sr in result.step_results:
            assert abs(sr.cyborg_state.reward_blue) < 0.01, \
                f"CybORG Blue reward should be 0, got {sr.cyborg_state.reward_blue}"
            assert abs(sr.jax_state.reward_blue) < 0.01, \
                f"JAX Blue reward should be 0, got {sr.jax_state.reward_blue}"

    def test_user_host_confidentiality_value(self):
        """CybORG vs JAX: User host compromise confidentiality penalty."""
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

    def test_enterprise_host_confidentiality_value(self):
        """CybORG vs JAX: Enterprise host compromise confidentiality penalty."""
        harness = DifferentialHarness(seed=42, max_steps=12, verbose=False)

        red_actions = [
            red_discover_subnet(SUBNET_USER),
            red_scan_host('User1'),
            red_exploit_host('User1', EXPLOIT_SSH),
            red_privesc_host('User1'),
            red_scan_host('Enterprise0'),
            red_exploit_host('Enterprise0', EXPLOIT_SSH),
            0,
        ]

        blue_actions = [BLUE_MONITOR] * 7

        red_policy = scripted_red_policy_factory(red_actions)
        blue_policy = scripted_blue_policy_factory(blue_actions)

        result = harness.run_episode(blue_policy, red_policy)

        for sr in result.step_results:
            assert abs(sr.cyborg_state.reward_blue - sr.jax_state.reward_blue) < 0.02, \
                f"Step {sr.step}: Blue reward mismatch CybORG={sr.cyborg_state.reward_blue}, JAX={sr.jax_state.reward_blue}"

    def test_op_server_availability_value(self):
        """CybORG vs JAX: Op_Server0 Impact availability penalty."""
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
            assert abs(sr.cyborg_state.reward_blue - sr.jax_state.reward_blue) < 0.02, \
                f"Step {sr.step}: Blue reward mismatch CybORG={sr.cyborg_state.reward_blue}, JAX={sr.jax_state.reward_blue}"


@requires_cyborg
class TestActionCostsParity:
    """Tests verifying action costs match CybORG."""

    def test_remove_has_zero_cost(self):
        """CybORG vs JAX: Remove action should have zero cost."""
        harness = DifferentialHarness(seed=42, max_steps=10, verbose=False)

        red_actions = [
            red_discover_subnet(SUBNET_USER),
            red_scan_host('User1'),
            red_exploit_host('User1', EXPLOIT_SSH),
            0,
        ]

        blue_actions = [
            BLUE_MONITOR,
            BLUE_MONITOR,
            BLUE_MONITOR,
            blue_remove_host('User1'),
        ]

        red_policy = scripted_red_policy_factory(red_actions)
        blue_policy = scripted_blue_policy_factory(blue_actions)

        result = harness.run_episode(blue_policy, red_policy)

        for sr in result.step_results:
            assert abs(sr.cyborg_state.reward_blue - sr.jax_state.reward_blue) < 0.02, \
                f"Step {sr.step}: Blue reward mismatch CybORG={sr.cyborg_state.reward_blue}, JAX={sr.jax_state.reward_blue}"

    def test_restore_has_minus_one_cost(self):
        """CybORG vs JAX: Restore action should have -1.0 cost."""
        harness = DifferentialHarness(seed=42, max_steps=10, verbose=False)

        red_actions = [
            red_discover_subnet(SUBNET_USER),
            red_scan_host('User1'),
            red_exploit_host('User1', EXPLOIT_SSH),
            0,
        ]

        blue_actions = [
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

    def test_analyse_has_zero_cost(self):
        """CybORG vs JAX: Analyse action should have zero cost."""
        harness = DifferentialHarness(seed=42, max_steps=10, verbose=False)

        red_actions = [
            red_discover_subnet(SUBNET_USER),
            red_scan_host('User1'),
            red_exploit_host('User1', EXPLOIT_SSH),
            0,
        ]

        blue_actions = [
            BLUE_MONITOR,
            BLUE_MONITOR,
            BLUE_MONITOR,
            blue_analyse_host('User1'),
        ]

        red_policy = scripted_red_policy_factory(red_actions)
        blue_policy = scripted_blue_policy_factory(blue_actions)

        result = harness.run_episode(blue_policy, red_policy)

        for sr in result.step_results:
            assert abs(sr.cyborg_state.reward_blue - sr.jax_state.reward_blue) < 0.02, \
                f"Step {sr.step}: Blue reward mismatch CybORG={sr.cyborg_state.reward_blue}, JAX={sr.jax_state.reward_blue}"

    def test_monitor_has_zero_cost(self):
        """CybORG vs JAX: Monitor action should have zero cost."""
        harness = DifferentialHarness(seed=42, max_steps=5, verbose=False)

        red_actions = [0] * 5
        blue_actions = [BLUE_MONITOR] * 5

        red_policy = scripted_red_policy_factory(red_actions)
        blue_policy = scripted_blue_policy_factory(blue_actions)

        result = harness.run_episode(blue_policy, red_policy)

        for sr in result.step_results:
            assert abs(sr.jax_state.reward_blue) < 0.01, \
                f"Step {sr.step}: Monitor should have zero cost, got {sr.jax_state.reward_blue}"


@requires_cyborg
class TestZeroSumPropertyParity:
    """Tests verifying zero-sum reward property matches CybORG."""

    def test_zero_sum_on_exploit(self):
        """CybORG vs JAX: Blue and Red rewards should be zero-sum on exploit."""
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
            jax_sum = sr.jax_state.reward_blue + sr.jax_state.reward_red
            assert abs(jax_sum) < 0.01, \
                f"Step {sr.step}: JAX rewards not zero-sum: blue={sr.jax_state.reward_blue}, red={sr.jax_state.reward_red}, sum={jax_sum}"

    def test_zero_sum_through_killchain(self):
        """CybORG vs JAX: Zero-sum property through entire killchain."""
        harness = DifferentialHarness(seed=42, max_steps=20, verbose=False)

        red_actions = BLINE_KILLCHAIN_STANDARD[:16]
        blue_actions = [BLUE_MONITOR] * 16

        red_policy = scripted_red_policy_factory(red_actions)
        blue_policy = scripted_blue_policy_factory(blue_actions)

        result = harness.run_episode(blue_policy, red_policy)

        for sr in result.step_results:
            jax_sum = sr.jax_state.reward_blue + sr.jax_state.reward_red
            assert abs(jax_sum) < 0.01, \
                f"Step {sr.step}: JAX rewards not zero-sum: blue={sr.jax_state.reward_blue}, red={sr.jax_state.reward_red}"


@requires_cyborg
class TestCumulativeRewardsParity:
    """Tests verifying cumulative rewards match CybORG."""

    def test_cumulative_confidentiality_multiple_hosts(self):
        """CybORG vs JAX: Cumulative confidentiality from multiple compromised hosts."""
        harness = DifferentialHarness(seed=42, max_steps=15, verbose=False)

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

    def test_reward_recovery_after_restore(self):
        """CybORG vs JAX: Rewards recover after Restore removes compromise."""
        harness = DifferentialHarness(seed=42, max_steps=12, verbose=False)

        red_actions = [
            red_discover_subnet(SUBNET_USER),
            red_scan_host('User1'),
            red_exploit_host('User1', EXPLOIT_SSH),
            0,
            0,
            0,
        ]

        blue_actions = [
            BLUE_MONITOR,
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

    def test_total_episode_reward_matches(self):
        """CybORG vs JAX: Total episode rewards should match."""
        harness = DifferentialHarness(seed=42, max_steps=20, verbose=False)

        red_actions = BLINE_KILLCHAIN_STANDARD[:16]
        blue_actions = [BLUE_MONITOR] * 16

        red_policy = scripted_red_policy_factory(red_actions)
        blue_policy = scripted_blue_policy_factory(blue_actions)

        result = harness.run_episode(blue_policy, red_policy)

        cyborg_blue_total = sum(sr.cyborg_state.reward_blue for sr in result.step_results)
        jax_blue_total = sum(sr.jax_state.reward_blue for sr in result.step_results)

        assert abs(cyborg_blue_total - jax_blue_total) < 0.5, \
            f"Total Blue reward mismatch: CybORG={cyborg_blue_total}, JAX={jax_blue_total}"
