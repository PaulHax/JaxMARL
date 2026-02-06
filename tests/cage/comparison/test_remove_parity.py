"""Differential tests comparing CybORG Remove behavior to JAX.

Every test runs both CybORG and JAX with identical actions and compares results.
These tests verify reward parity for the Remove action.
"""

import pytest
from tests.cage.differential.harness import DifferentialHarness, is_cyborg_available
from tests.cage.differential.state_comparator import StateSnapshot, COMPROMISE_USER, COMPROMISE_PRIVILEGED
from tests.cage.comparison.policies import (
    scripted_blue_policy_factory,
    scripted_red_policy_factory,
    sleep_policy,
    monitor_policy,
)
from tests.cage.comparison.scenarios import (
    red_discover_subnet,
    red_scan_host,
    red_exploit_host,
    red_privesc_host,
    blue_remove_host,
    blue_analyse_host,
    SUBNET_USER,
    EXPLOIT_SSH,
)
from jaxmarl.environments.cage.actions import BLUE_SLEEP, BLUE_MONITOR
from jaxmarl.environments.cage.state import HOST_IDS, ACTIVITY_EXPLOIT


requires_cyborg = pytest.mark.skipif(
    not is_cyborg_available(),
    reason="CybORG not installed"
)


@requires_cyborg
class TestRemovePenaltyCybORGParity:
    """Differential tests comparing CybORG Remove behavior to JAX.

    Key CybORG behavior:
    - Remove requires prior activity detection (Monitor/Analyse must run after compromise)
    - Remove only removes USER-level sessions (not PRIVILEGED/root)
    - Remove does NOT have an action penalty (unlike Restore)
    - Blue acts FIRST each step, so Monitor must run in a LATER step than Exploit
    """

    def test_remove_on_user_compromise_no_penalty(self):
        """CybORG vs JAX: Remove on USER-level compromised host should match rewards.

        IMPORTANT: Monitor after SSH exploit sees connection anomalies, but
        SSHBruteForce does NOT create actionable PIDs in CybORG. Remove fails
        and user access persists.
        """
        harness = DifferentialHarness(seed=42, max_steps=10, verbose=False)

        red_actions = [
            red_discover_subnet(SUBNET_USER),
            red_scan_host('User1'),
            red_exploit_host('User1', EXPLOIT_SSH),
            0,  # Sleep - stay at USER level
            0,  # Sleep
        ]

        blue_actions = [
            BLUE_MONITOR,
            BLUE_MONITOR,
            BLUE_MONITOR,
            BLUE_MONITOR,  # Monitor AFTER Exploit to detect activity
            blue_remove_host('User1'),  # Remove at step 4 when activity has been detected
        ]

        red_policy = scripted_red_policy_factory(red_actions)
        blue_policy = scripted_blue_policy_factory(blue_actions)

        result = harness.run_episode(blue_policy, red_policy)

        remove_step = result.step_results[4]  # Step 5 (0-indexed as 4)

        assert abs(remove_step.cyborg_state.reward_blue - remove_step.jax_state.reward_blue) < 0.02, \
            f"Blue reward mismatch at Remove step: CybORG={remove_step.cyborg_state.reward_blue}, JAX={remove_step.jax_state.reward_blue}"

        assert remove_step.jax_state.red_privilege.get('User1', 0) == 1, \
            "SSH exploit does not create actionable PIDs in CybORG; Remove should fail and leave user access"

    def test_ssh_exploit_observed_but_not_actionable(self):
        """SSH exploit should be observed (activity bits), but Remove still fails."""
        harness = DifferentialHarness(seed=42, max_steps=10, verbose=False, check_obs=True, sync_detection_rng=True)

        red_actions = [
            red_discover_subnet(SUBNET_USER),
            red_scan_host('User1'),
            red_exploit_host('User1', EXPLOIT_SSH),
            0,  # Sleep
            0,  # Sleep
        ]

        blue_actions = [
            BLUE_MONITOR,
            BLUE_MONITOR,
            BLUE_MONITOR,
            BLUE_MONITOR,  # Monitor AFTER Exploit to detect activity
            blue_remove_host('User1'),
        ]

        red_policy = scripted_red_policy_factory(red_actions)
        blue_policy = scripted_blue_policy_factory(blue_actions)

        harness.reset()
        # Run through exploit step
        for step in range(3):
            harness.step(blue_actions[step], red_actions[step])

        jax_user1 = HOST_IDS['User1']
        assert int(harness.jax_state.red_activity_this_step[jax_user1]) == ACTIVITY_EXPLOIT, \
            "JAX should mark exploit activity for User1 during SSH brute force"

        # Monitor after exploit should record observation but no actionable PID
        harness.step(blue_actions[3], red_actions[3])
        assert bool(harness.jax_state.host_activity_detected[jax_user1]), \
            "JAX should record observed activity for User1 after SSH brute force"
        assert not bool(harness.jax_state.host_activity_actionable[jax_user1]), \
            "SSH exploit should not create actionable PIDs for Remove"

        # Remove should still fail (CybORG parity)
        remove_step = harness.step(blue_actions[4], red_actions[4])
        assert remove_step.jax_state.red_privilege.get('User1', 0) == 1

    def test_remove_on_privileged_compromise_no_extra_penalty(self):
        """CybORG vs JAX: Remove on PRIVILEGED host - verify no extra penalty.

        This is the key test case from the investigation. CybORG Remove doesn't
        apply extra penalties based on privilege level. Remove simply attempts
        to kill user-level sessions.
        """
        harness = DifferentialHarness(seed=42, max_steps=10, verbose=False)

        red_actions = [
            red_discover_subnet(SUBNET_USER),
            red_scan_host('User1'),
            red_exploit_host('User1', EXPLOIT_SSH),
            red_privesc_host('User1'),
            0,  # Sleep
        ]

        blue_actions = [
            BLUE_MONITOR,
            BLUE_MONITOR,
            BLUE_MONITOR,
            BLUE_MONITOR,
            blue_remove_host('User1'),
        ]

        red_policy = scripted_red_policy_factory(red_actions)
        blue_policy = scripted_blue_policy_factory(blue_actions)

        result = harness.run_episode(blue_policy, red_policy)

        remove_step = result.step_results[4]

        assert abs(remove_step.cyborg_state.reward_blue - remove_step.jax_state.reward_blue) < 0.02, \
            f"Blue reward mismatch at Remove step: CybORG={remove_step.cyborg_state.reward_blue}, JAX={remove_step.jax_state.reward_blue}"

    def test_remove_on_clean_host_no_penalty(self):
        """CybORG vs JAX: Remove on non-compromised host should not penalize."""
        harness = DifferentialHarness(seed=42, max_steps=5, verbose=False)

        red_actions = [
            0,  # Sleep
            0,  # Sleep
        ]

        blue_actions = [
            BLUE_MONITOR,
            blue_remove_host('Enterprise0'),  # Remove on clean host
        ]

        red_policy = scripted_red_policy_factory(red_actions)
        blue_policy = scripted_blue_policy_factory(blue_actions)

        result = harness.run_episode(blue_policy, red_policy)

        remove_step = result.step_results[1]  # Step 2 (0-indexed as 1)

        assert abs(remove_step.cyborg_state.reward_blue - remove_step.jax_state.reward_blue) < 0.02, \
            f"Blue reward mismatch on clean host Remove: CybORG={remove_step.cyborg_state.reward_blue}, JAX={remove_step.jax_state.reward_blue}"

    def test_remove_vs_red_privesc_race(self):
        """CybORG vs JAX: Remove and PrivEsc happening simultaneously."""
        harness = DifferentialHarness(seed=42, max_steps=10, verbose=False)

        red_actions = [
            red_discover_subnet(SUBNET_USER),
            red_scan_host('User2'),
            red_exploit_host('User2', EXPLOIT_SSH),
            red_privesc_host('User2'),
            0,
        ]

        blue_actions = [
            BLUE_MONITOR,
            BLUE_MONITOR,
            BLUE_MONITOR,
            blue_remove_host('User2'),
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

    def test_remove_after_analyse_detects_activity(self):
        """CybORG vs JAX: Remove after Analyse has detected activity."""
        harness = DifferentialHarness(seed=42, max_steps=10, verbose=False)

        red_actions = [
            red_discover_subnet(SUBNET_USER),
            red_scan_host('User2'),
            red_exploit_host('User2', EXPLOIT_SSH),
            0,  # Sleep
            0,  # Sleep
        ]

        blue_actions = [
            BLUE_MONITOR,
            BLUE_MONITOR,
            BLUE_MONITOR,
            blue_analyse_host('User2'),  # Analyse to detect
            blue_remove_host('User2'),   # Then remove
        ]

        red_policy = scripted_red_policy_factory(red_actions)
        blue_policy = scripted_blue_policy_factory(blue_actions)

        result = harness.run_episode(blue_policy, red_policy)

        for sr in result.step_results:
            cyborg_blue = sr.cyborg_state.reward_blue
            jax_blue = sr.jax_state.reward_blue
            assert abs(cyborg_blue - jax_blue) < 0.02, \
                f"Step {sr.step}: Blue reward mismatch CybORG={cyborg_blue}, JAX={jax_blue}"

    @pytest.mark.parametrize("host", ['User1', 'User2', 'User3', 'User4'])
    def test_remove_on_different_user_hosts(self, host):
        """CybORG vs JAX: Remove behavior should match across different hosts."""
        harness = DifferentialHarness(seed=42, max_steps=8, verbose=False)

        red_actions = [
            red_discover_subnet(SUBNET_USER),
            red_scan_host(host),
            red_exploit_host(host, EXPLOIT_SSH),
            0,  # Sleep
        ]

        blue_actions = [
            BLUE_MONITOR,
            BLUE_MONITOR,
            BLUE_MONITOR,
            blue_remove_host(host),
        ]

        red_policy = scripted_red_policy_factory(red_actions)
        blue_policy = scripted_blue_policy_factory(blue_actions)

        result = harness.run_episode(blue_policy, red_policy)

        remove_step = result.step_results[3]
        assert abs(remove_step.cyborg_state.reward_blue - remove_step.jax_state.reward_blue) < 0.02, \
            f"Remove on {host}: Blue reward mismatch CybORG={remove_step.cyborg_state.reward_blue}, JAX={remove_step.jax_state.reward_blue}"

    def test_multiple_removes_in_episode(self):
        """CybORG vs JAX: Multiple Remove actions in one episode."""
        harness = DifferentialHarness(seed=42, max_steps=12, verbose=False)

        red_actions = [
            red_discover_subnet(SUBNET_USER),
            red_scan_host('User1'),
            red_exploit_host('User1', EXPLOIT_SSH),
            red_scan_host('User2'),
            red_exploit_host('User2', EXPLOIT_SSH),
            0,
            0,
        ]

        blue_actions = [
            BLUE_MONITOR,
            BLUE_MONITOR,
            BLUE_MONITOR,
            BLUE_MONITOR,
            BLUE_MONITOR,
            blue_remove_host('User1'),
            blue_remove_host('User2'),
        ]

        red_policy = scripted_red_policy_factory(red_actions)
        blue_policy = scripted_blue_policy_factory(blue_actions)

        result = harness.run_episode(blue_policy, red_policy)

        for sr in result.step_results:
            cyborg_blue = sr.cyborg_state.reward_blue
            jax_blue = sr.jax_state.reward_blue
            assert abs(cyborg_blue - jax_blue) < 0.02, \
                f"Step {sr.step}: Blue reward mismatch CybORG={cyborg_blue}, JAX={jax_blue}"
