"""Differential tests comparing CybORG observation encoding to JAX.

Every test runs both CybORG and JAX with identical actions and compares
Blue agent observations after various scenarios.
"""

import pytest
import random
import numpy as np
import jax
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
    EXPLOIT_ETERNAL,
    BLINE_KILLCHAIN_STANDARD,
)
from jaxmarl.environments.cage.actions import BLUE_MONITOR
from jaxmarl.environments.cage.state import HOST_IDS


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

    def test_user2_remove_observation_parity_without_detection(self, monkeypatch):
        """CybORG vs JAX: User2 observation parity when exploit detection is disabled."""
        from CybORG import CybORG
        from CybORG.Agents.Wrappers import BlueTableWrapper
        from CybORG.Shared.Actions.ConcreteActions.ExploitAction import ExploitAction
        from tests.cage.differential.action_translator import jax_action_to_cyborg
        from jaxmarl.environments.cage import CageEnv
        import jaxmarl.environments.cage.actions as jax_actions
        import inspect

        # Force exploit detection to fail in both envs for deterministic parity
        orig_init = ExploitAction.__init__

        def patched_init(self, *args, **kwargs):
            orig_init(self, *args, **kwargs)
            self.detection_rate = 0.0

        monkeypatch.setattr(ExploitAction, "__init__", patched_init)
        monkeypatch.setattr(jax_actions, "EXPLOIT_DETECTION_RATE", 0.0)

        seed = 123
        random.seed(seed)

        path = str(inspect.getfile(CybORG))
        path = path[:-10] + "/Shared/Scenarios/Scenario2.yaml"
        cyborg = CybORG(path, "sim")
        cyborg.reset()
        cyborg.set_seed(seed)

        blue_wrapper = BlueTableWrapper(env=cyborg, output_mode="vector")
        blue_wrapper.reset("Blue")

        with jax.disable_jit():
            jax_env = CageEnv()
            key = jax.random.PRNGKey(seed)
            obs_jax, state = jax_env.reset(key)

            red_actions = [
                red_discover_subnet(SUBNET_USER),
                red_scan_host("User2"),
                red_exploit_host("User2", EXPLOIT_ETERNAL),
            ]

            for red_action in red_actions:
                cyborg.step("Blue", jax_action_to_cyborg(0, cyborg, "Blue"))
                cyborg.step("Red", jax_action_to_cyborg(red_action, cyborg, "Red"))

                key, subkey = jax.random.split(key)
                obs_jax, state, _, _, _ = jax_env.step_env(
                    subkey, state, {"blue": 0, "red": red_action}
                )

            blue_remove = blue_remove_host("User2")
            cyborg.step("Blue", jax_action_to_cyborg(blue_remove, cyborg, "Blue"))
            cyborg.step("Red", jax_action_to_cyborg(0, cyborg, "Red"))
            obs_cyb = blue_wrapper.observation_change(cyborg.get_observation("Blue"))

            key, subkey = jax.random.split(key)
            obs_jax, state, _, _, _ = jax_env.step_env(
                subkey, state, {"blue": blue_remove, "red": 0}
            )

            # Compare User2 activity/compromise bits after Remove
            base = HOST_IDS["User2"] * 4
            cyb_user2 = obs_cyb[base:base + 4]
            jax_user2 = np.array(obs_jax["blue"][base:base + 4])

            assert np.allclose(cyb_user2, jax_user2), (
                f"User2 obs mismatch after Remove: CybORG={cyb_user2}, JAX={jax_user2}"
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
