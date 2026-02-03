"""Multi-step trajectory comparison tests between CybORG and JAXmarl CAGE.

Tests complex attack patterns, blue defense strategies, and full episode
equivalence across multiple seeds and agents.
"""

import pytest
import numpy as np

from tests.cage.differential.harness import (
    DifferentialHarness, is_cyborg_available, sleep_policy, monitor_policy,
)
from tests.cage.differential.state_comparator import StateSnapshot
from tests.cage.comparison.policies import (
    react_remove_policy_with_timing, react_restore_policy_with_timing,
    decoy_defense_policy,
)
from tests.cage.comparison.scenarios import DECOY_SSHD


requires_cyborg = pytest.mark.skipif(
    not is_cyborg_available(),
    reason="CybORG not installed"
)


@requires_cyborg
class TestBLineAgentTrajectories:
    """Test B_lineAgent trajectory comparison."""

    @pytest.mark.parametrize("seed", [42, 123, 456, 789, 1234])
    def test_bline_trajectory_rewards_match(self, seed):
        """B_line agent rewards should match at each step across seeds."""
        harness = DifferentialHarness(seed=seed, max_steps=50, verbose=False)

        result = harness.run_bline_episode(sleep_policy, use_jax_bline=False)

        assert result.steps_completed == 50

        for sr in result.step_results:
            cyborg_blue = sr.cyborg_state.reward_blue
            jax_blue = sr.jax_state.reward_blue
            cyborg_red = sr.cyborg_state.reward_red
            jax_red = sr.jax_state.reward_red

            assert abs(cyborg_blue - jax_blue) <= 0.1, \
                f"Step {sr.step}: blue reward mismatch - CybORG={cyborg_blue}, JAX={jax_blue}"
            assert abs(cyborg_red - jax_red) <= 0.05, \
                f"Step {sr.step}: red reward mismatch - CybORG={cyborg_red}, JAX={jax_red}"

    @pytest.mark.parametrize("seed", [42, 100, 200])
    def test_bline_state_consistency(self, seed):
        """B_line agent should produce consistent state transitions."""
        harness = DifferentialHarness(seed=seed, max_steps=30, verbose=False)

        result = harness.run_bline_episode(sleep_policy, use_jax_bline=False)

        assert result.error_diffs == 0, \
            f"Found {result.error_diffs} state differences"


@requires_cyborg
class TestKillchainTrajectories:
    """Test killchain trajectories via B_line agent."""

    @pytest.fixture
    def harness(self):
        return DifferentialHarness(seed=42, max_steps=30, verbose=False)

    def test_bline_reaches_impact(self, harness):
        """B_line should reach Impact action."""
        result = harness.run_bline_episode(sleep_policy, use_jax_bline=False)

        impact_reached = any('Impact' in sr.red_action_desc for sr in result.step_results)
        assert impact_reached, "Should reach Impact action"

        assert result.error_diffs == 0, f"State mismatch: {result.failure_reason}"

    def test_killchain_state_matches(self, harness):
        """Killchain state should match throughout trajectory."""
        result = harness.run_bline_episode(sleep_policy, use_jax_bline=False)

        for sr in result.step_results:
            for hostname in sr.jax_state.host_compromised:
                cyborg_comp = sr.cyborg_state.host_compromised.get(hostname, 0)
                jax_comp = sr.jax_state.host_compromised.get(hostname, 0)
                assert cyborg_comp == jax_comp, \
                    f"Step {sr.step}: {hostname} compromise mismatch"


@requires_cyborg
class TestBlueStrategyVariations:
    """Test different Blue defense strategies."""

    @pytest.fixture
    def harness(self):
        return DifferentialHarness(seed=42, max_steps=50, verbose=False)

    def test_react_remove_strategy(self, harness):
        """ReactRemove strategy should match CybORG behavior."""
        remove_policy = react_remove_policy_with_timing(delay=0)
        result = harness.run_bline_episode(remove_policy, use_jax_bline=False)

        assert result.steps_completed == 50
        assert result.error_diffs == 0, f"State mismatch: {result.failure_reason}"

    def test_react_remove_with_delay(self, harness):
        """ReactRemove with 1-step delay should match."""
        remove_policy = react_remove_policy_with_timing(delay=1)
        result = harness.run_bline_episode(remove_policy, use_jax_bline=False)

        assert result.steps_completed == 50
        assert result.error_diffs == 0, f"State mismatch: {result.failure_reason}"

    def test_react_restore_strategy(self, harness):
        """ReactRestore strategy should match CybORG behavior."""
        restore_policy = react_restore_policy_with_timing(delay=0)
        result = harness.run_bline_episode(restore_policy, use_jax_bline=False)

        assert result.error_diffs == 0, f"State mismatch: {result.failure_reason}"

    def test_decoy_defense_strategy(self, harness):
        """Decoy deployment strategy should match."""
        placements = [
            (0, 'Enterprise0', DECOY_SSHD),
            (1, 'Enterprise1', DECOY_SSHD),
            (2, 'Enterprise2', DECOY_SSHD),
        ]
        decoy_policy = decoy_defense_policy(placements)
        result = harness.run_bline_episode(decoy_policy, use_jax_bline=False)

        assert result.steps_completed == 50
        assert result.error_diffs == 0, f"State mismatch: {result.failure_reason}"


@requires_cyborg
class TestFullEpisodeEquivalence:
    """Test full 100-step episode equivalence with parametrized agents/seeds."""

    @pytest.mark.parametrize("seed", [42, 123, 456])
    def test_100_step_passive_blue(self, seed):
        """Full episode with passive Blue should match."""
        harness = DifferentialHarness(seed=seed, max_steps=100, verbose=False)
        result = harness.run_bline_episode(sleep_policy, use_jax_bline=False)

        assert result.steps_completed == 100, f"Only completed {result.steps_completed} steps"
        assert result.error_diffs == 0, f"Seed {seed}: {result.error_diffs} differences"

    @pytest.mark.parametrize("seed", [42, 123, 456])
    def test_100_step_monitor_blue(self, seed):
        """Full episode with monitoring Blue should match."""
        harness = DifferentialHarness(seed=seed, max_steps=100, verbose=False)
        result = harness.run_bline_episode(monitor_policy, use_jax_bline=False)

        assert result.steps_completed == 100
        assert result.error_diffs == 0, f"Seed {seed}: {result.error_diffs} differences"

    @pytest.mark.parametrize("seed", [10, 20, 30, 40, 50])
    def test_cumulative_reward_equivalence(self, seed):
        """Cumulative rewards should match within tolerance."""
        harness = DifferentialHarness(seed=seed, max_steps=100, verbose=False)
        result = harness.run_bline_episode(sleep_policy, use_jax_bline=False)

        cyborg_blue, cyborg_red = result.cyborg_total_reward
        jax_blue, jax_red = result.jax_total_reward

        assert abs(cyborg_blue - jax_blue) < 1.0, \
            f"Seed {seed}: Blue reward diff too large: CybORG={cyborg_blue}, JAX={jax_blue}"
        assert abs(cyborg_red - jax_red) < 1.0, \
            f"Seed {seed}: Red reward diff too large: CybORG={cyborg_red}, JAX={jax_red}"

    def test_determinism_same_seed(self):
        """Same seed should produce identical trajectories."""
        seed = 42

        harness1 = DifferentialHarness(seed=seed, max_steps=50, verbose=False)
        result1 = harness1.run_bline_episode(sleep_policy, use_jax_bline=False)

        harness2 = DifferentialHarness(seed=seed, max_steps=50, verbose=False)
        result2 = harness2.run_bline_episode(sleep_policy, use_jax_bline=False)

        for sr1, sr2 in zip(result1.step_results, result2.step_results):
            assert sr1.red_action_jax == sr2.red_action_jax, \
                f"Step {sr1.step}: actions differ"
            assert abs(sr1.jax_state.reward_red - sr2.jax_state.reward_red) < 0.01


@requires_cyborg
class TestRewardAccumulation:
    """Test reward accumulation matches step by step."""

    @pytest.fixture
    def harness(self):
        return DifferentialHarness(seed=42, max_steps=100, verbose=False)

    def test_step_rewards_accumulate(self, harness):
        """Step rewards should accumulate correctly."""
        result = harness.run_bline_episode(sleep_policy, use_jax_bline=False)

        for i, sr in enumerate(result.step_results):
            cyborg_blue = sr.cyborg_state.reward_blue
            jax_blue = sr.jax_state.reward_blue
            cyborg_red = sr.cyborg_state.reward_red
            jax_red = sr.jax_state.reward_red

            assert abs(cyborg_blue - jax_blue) < 0.1, \
                f"Step {sr.step}: Blue reward mismatch"
            assert abs(cyborg_red - jax_red) < 0.05, \
                f"Step {sr.step}: Red reward mismatch"

    def test_no_reward_drift(self, harness):
        """Rewards should not drift over time."""
        result = harness.run_bline_episode(sleep_policy, use_jax_bline=False)

        max_blue_diff = 0.0
        max_red_diff = 0.0

        for sr in result.step_results:
            blue_diff = abs(sr.cyborg_state.reward_blue - sr.jax_state.reward_blue)
            red_diff = abs(sr.cyborg_state.reward_red - sr.jax_state.reward_red)
            max_blue_diff = max(max_blue_diff, blue_diff)
            max_red_diff = max(max_red_diff, red_diff)

        assert max_blue_diff < 0.2, f"Blue reward drift too high: {max_blue_diff}"
        assert max_red_diff < 0.1, f"Red reward drift too high: {max_red_diff}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
