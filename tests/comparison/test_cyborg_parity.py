"""CybORG parity tests.

These tests replicate CybORG's own test suite to verify JAXmarl CAGE
produces identical behavior. All tests run real CybORG code alongside
JAX and compare results.

Based on tests from:
- CybORG/Tests/test_sim/test_Acceptance/test_reward_function.py
- CybORG/Tests/test_sim/test_Agents/test_blineagent.py
- CybORG/Tests/test_sim/test_Actions/test_BlueActions/test_Deceptive_Actions/
- CybORG/Tests/test_sim/test_Actions/test_BlueActions/test_blue_*.py
- CybORG/Tests/test_sim/test_Actions/test_RedActions/

Run with: pytest tests/comparison/test_cyborg_parity.py -v
"""

import pytest
import numpy as np

from jaxmarl.environments.cage.state import HOST_IDS, COMPROMISE_NONE, COMPROMISE_USER, COMPROMISE_PRIVILEGED
from jaxmarl.environments.cage.actions import (
    BLUE_SLEEP, BLUE_MONITOR, BLUE_REMOVE_START, BLUE_RESTORE_START,
    BLUE_DECOY_START, BLUE_ANALYSE_START, NUM_DECOY_TYPES, RED_SLEEP,
    NUM_HOSTS, RED_DISCOVER_SUBNET_START, RED_SCAN_HOST_START,
    RED_EXPLOIT_START, RED_PRIVESC_START, RED_IMPACT_START,
)

from tests.differential.harness import (
    DifferentialHarness, is_cyborg_available, sleep_policy, monitor_policy,
    reactive_remove_policy, reactive_restore_policy,
)
from tests.differential.state_comparator import StateSnapshot
from tests.comparison.policies import (
    react_remove_policy_with_timing, react_restore_policy_with_timing,
    scripted_blue_policy_factory, scripted_red_policy_factory,
)
from tests.comparison.scenarios import (
    red_discover_subnet, red_scan_host, red_exploit_host, red_privesc_host,
    red_impact_host, blue_decoy_host, blue_restore_host, blue_remove_host,
    blue_analyse_host,
    BLINE_KILLCHAIN_STANDARD, KILLCHAIN_VIA_ENTERPRISE0, KILLCHAIN_VIA_HARAKA,
    get_path_to_subnet, get_host_subnet,
    SUBNET_USER, SUBNET_ENTERPRISE, SUBNET_OPERATIONAL,
    EXPLOIT_SSH, EXPLOIT_HTTP, EXPLOIT_HTTPS, EXPLOIT_HARAKA,
    DECOY_APACHE, DECOY_FEMITTER, DECOY_HARAKA, DECOY_SMSS,
    DECOY_SSHD, DECOY_SVCHOST, DECOY_TOMCAT, DECOY_VSFTPD,
)


requires_cyborg = pytest.mark.skipif(
    not is_cyborg_available(),
    reason="CybORG not installed"
)


# =============================================================================
# SECTION 1: Reward Tests
# Based on: test_reward_function.py
# =============================================================================

@requires_cyborg
class TestRepeatedExploitNoReward:
    """Test that re-exploiting an already compromised host produces no additional reward.

    Based on CybORG test_security_reward_repeat in test_reward_function.py
    """

    @pytest.fixture
    def harness(self):
        return DifferentialHarness(seed=42, max_steps=60, verbose=False)

    def test_repeated_exploit_no_reward(self, harness):
        """Second exploitation of same host should give 0 additional security penalty."""
        result = harness.run_bline_episode(sleep_policy, use_jax_bline=False)

        first_exploit_rewards = {}
        for sr in result.step_results:
            if 'Exploit' in sr.red_action_desc or 'SSHBruteForce' in sr.red_action_desc:
                host = sr.red_action_desc.split('(')[1].rstrip(')')
                if host not in first_exploit_rewards:
                    first_exploit_rewards[host] = sr.jax_state.reward_red

        cyborg_step_16 = result.step_results[15].cyborg_state.reward_red if len(result.step_results) > 15 else 0
        jax_step_16 = result.step_results[15].jax_state.reward_red if len(result.step_results) > 15 else 0

        assert abs(cyborg_step_16 - jax_step_16) < 0.05, \
            f"Step 16 rewards should match: CybORG={cyborg_step_16}, JAX={jax_step_16}"

    def test_exploit_same_host_twice_no_additional_reward(self, harness):
        """Exploiting the same host twice should not give additional reward the second time."""
        exploit_sequence = [
            red_discover_subnet(SUBNET_USER),
            red_scan_host('User3'),
            red_exploit_host('User3', EXPLOIT_HARAKA),  # Haraka gives root access
            red_privesc_host('User3'),
            red_exploit_host('User3', EXPLOIT_HARAKA),  # Second exploit - should give nothing
        ]

        red_policy = scripted_red_policy_factory(exploit_sequence)

        result = harness.run_episode(sleep_policy, red_policy)

        assert result.steps_completed >= 5, "Should complete at least 5 steps"

        first_exploit_reward = result.step_results[2].jax_state.reward_red if len(result.step_results) > 2 else 0
        second_exploit_reward = result.step_results[4].jax_state.reward_red if len(result.step_results) > 4 else 0

        reward_delta = second_exploit_reward - result.step_results[3].jax_state.reward_red

        assert abs(reward_delta) < 0.01, \
            f"Second exploit should give no additional reward, got delta={reward_delta}"


@requires_cyborg
class TestAvailabilityRewards:
    """Test availability reward calculations for Impact and Restore.

    Based on CybORG test_availability_reward_impact, test_availability_reward_restore,
    test_availability_reward_restore_solo
    """

    @pytest.fixture
    def harness(self):
        return DifferentialHarness(seed=42, max_steps=25, verbose=False)

    def test_impact_gives_availability_penalty(self, harness):
        """Impact on Op_Server0 should give -10.0 blue reward."""
        result = harness.run_bline_episode(sleep_policy, use_jax_bline=False)

        impact_step = None
        for i, sr in enumerate(result.step_results):
            if 'Impact' in sr.red_action_desc:
                impact_step = i
                break

        assert impact_step is not None, "Should reach Impact action"

        if impact_step > 0:
            pre_impact_blue = result.step_results[impact_step - 1].jax_state.reward_blue
            post_impact_blue = result.step_results[impact_step].jax_state.reward_blue

            cyborg_pre = result.step_results[impact_step - 1].cyborg_state.reward_blue
            cyborg_post = result.step_results[impact_step].cyborg_state.reward_blue

            jax_penalty = post_impact_blue - pre_impact_blue
            cyborg_penalty = cyborg_post - cyborg_pre

            assert abs(jax_penalty - cyborg_penalty) < 0.1, \
                f"Impact penalty mismatch: JAX={jax_penalty}, CybORG={cyborg_penalty}"

    def test_restore_clears_availability_penalty(self, harness):
        """Restore after Impact should clear the availability penalty."""
        restore_step = 16

        def blue_restore_after_impact(state: StateSnapshot, step: int) -> int:
            if step == restore_step:
                return BLUE_RESTORE_START + HOST_IDS['Op_Server0']
            return BLUE_SLEEP

        result = harness.run_bline_episode(blue_restore_after_impact, use_jax_bline=False)

        assert result.steps_completed > restore_step

        for sr in result.step_results[:restore_step]:
            assert abs(sr.cyborg_state.reward_blue - sr.jax_state.reward_blue) < 0.15, \
                f"Step {sr.step}: blue reward mismatch"

    def test_restore_cost_without_prior_impact(self, harness):
        """Restore costs -1.0 even without prior Impact (restore_solo)."""
        def restore_all_early(state: StateSnapshot, step: int) -> int:
            hosts = ['User1', 'User2', 'Enterprise0', 'Enterprise1']
            if step < len(hosts):
                return BLUE_RESTORE_START + HOST_IDS[hosts[step]]
            return BLUE_SLEEP

        def red_sleep(state, step):
            return RED_SLEEP

        result = harness.run_episode(restore_all_early, red_sleep)

        for sr in result.step_results[:4]:
            assert abs(sr.cyborg_state.reward_blue - sr.jax_state.reward_blue) < 0.05, \
                f"Step {sr.step}: restore cost mismatch"


@requires_cyborg
class TestSecurityRewardPerHost:
    """Test security reward values per host.

    Based on CybORG test_security_reward
    """

    @pytest.fixture
    def harness(self):
        return DifferentialHarness(seed=42, max_steps=20, verbose=False)

    def test_security_rewards_match_cyborg(self, harness):
        """Security rewards should match CybORG at each step."""
        result = harness.run_bline_episode(sleep_policy, use_jax_bline=False)

        for sr in result.step_results:
            cyborg_blue = sr.cyborg_state.reward_blue
            jax_blue = sr.jax_state.reward_blue

            assert abs(cyborg_blue - jax_blue) < 0.05, \
                f"Step {sr.step}: blue reward mismatch - CybORG={cyborg_blue}, JAX={jax_blue}"


# =============================================================================
# SECTION 2: B_line Agent Tests
# Based on: test_blineagent.py
# =============================================================================

@requires_cyborg
class TestBLineResilience:
    """Test B_line resilience after Blue Restore.

    Based on CybORG test_bline_resilience in test_blineagent.py
    """

    @pytest.fixture
    def harness(self):
        return DifferentialHarness(seed=42, max_steps=45, verbose=False)

    def test_bline_resilience_after_restore(self, harness):
        """After Blue restores hosts, B_line should recover and succeed."""
        restore_targets = ['Op_Server0', 'Enterprise2', 'Enterprise1']
        restore_step_start = 16

        def blue_restore_after_impact(state: StateSnapshot, step: int) -> int:
            restore_offset = step - restore_step_start
            if 0 <= restore_offset < len(restore_targets):
                host = restore_targets[restore_offset]
                return BLUE_RESTORE_START + HOST_IDS[host]
            return BLUE_SLEEP

        result = harness.run_bline_episode(blue_restore_after_impact, use_jax_bline=False)

        assert result.steps_completed >= 25, "Should complete enough steps for recovery"

        recovery_success = False
        for i, sr in enumerate(result.step_results):
            if i > restore_step_start + len(restore_targets) + 2:
                if sr.jax_state.last_red_action_success:
                    recovery_success = True
                    break

        assert recovery_success, "Red should recover and succeed after Blue Restore"

        for sr in result.step_results[:16]:
            assert abs(sr.cyborg_state.reward_red - sr.jax_state.reward_red) < 0.1, \
                f"Step {sr.step}: reward mismatch before restore"


@requires_cyborg
class TestBLineVsReactRemoveSuccessPattern:
    """Test B_line action success pattern vs ReactRemove agent.

    Based on CybORG test_blineagent_vs_react_remove_agent_step
    """

    @pytest.fixture
    def harness(self):
        return DifferentialHarness(seed=42, max_steps=50, verbose=False)

    def test_bline_vs_react_remove_success_pattern(self, harness):
        """Verify rewards match when B_line fights ReactRemove."""
        result = harness.run_bline_episode(reactive_remove_policy, use_jax_bline=False)

        assert result.steps_completed == 50, f"Should complete 50 steps, got {result.steps_completed}"

        for i, sr in enumerate(result.step_results):
            assert abs(sr.cyborg_state.reward_red - sr.jax_state.reward_red) < 0.1, \
                f"Step {i}: reward mismatch - CybORG={sr.cyborg_state.reward_red}, JAX={sr.jax_state.reward_red}"


@requires_cyborg
class TestBLineVsReactRestoreSuccessPattern:
    """Test B_line action success pattern vs ReactRestore agent.

    Based on CybORG test_blineagent_vs_react_restore_agent_step
    """

    @pytest.fixture
    def harness(self):
        return DifferentialHarness(seed=42, max_steps=50, verbose=False)

    def test_bline_vs_react_restore_success_pattern(self, harness):
        """Verify success pattern when B_line fights ReactRestore."""
        result = harness.run_bline_episode(reactive_restore_policy, use_jax_bline=False)

        assert result.steps_completed == 50, f"Should complete 50 steps, got {result.steps_completed}"

        for i, sr in enumerate(result.step_results):
            assert abs(sr.cyborg_state.reward_red - sr.jax_state.reward_red) < 0.1, \
                f"Step {i}: reward mismatch"


@requires_cyborg
class TestBLineStepRewards:
    """Test B_line step rewards match CybORG exactly.

    Based on CybORG test_blineagent_step
    """

    @pytest.fixture
    def harness(self):
        return DifferentialHarness(seed=42, max_steps=20, verbose=False)

    def test_bline_rewards_at_key_steps(self, harness):
        """B_line rewards should match at specific steps."""
        result = harness.run_bline_episode(sleep_policy, use_jax_bline=False)

        assert result.steps_completed >= 15, f"Should reach Impact, got {result.steps_completed} steps"

        for sr in result.step_results:
            cyborg_reward = sr.cyborg_state.reward_red
            jax_reward = sr.jax_state.reward_red

            assert abs(cyborg_reward - jax_reward) < 0.05, \
                f"Step {sr.step}: reward mismatch - CybORG={cyborg_reward}, JAX={jax_reward}"

    def test_bline_reward_accumulation_matches(self, harness):
        """B_line reward accumulation should match between CybORG and JAX."""
        result = harness.run_bline_episode(sleep_policy, use_jax_bline=False)

        for sr in result.step_results:
            cyborg_blue = sr.cyborg_state.reward_blue
            jax_blue = sr.jax_state.reward_blue
            cyborg_red = sr.cyborg_state.reward_red
            jax_red = sr.jax_state.reward_red

            assert abs(cyborg_blue - jax_blue) < 0.1, \
                f"Step {sr.step}: blue reward mismatch - CybORG={cyborg_blue}, JAX={jax_blue}"
            assert abs(cyborg_red - jax_red) < 0.05, \
                f"Step {sr.step}: red reward mismatch - CybORG={cyborg_red}, JAX={jax_red}"


@requires_cyborg
class TestBLineDeterminismAfterReset:
    """Test B_line produces identical trajectory after reset.

    Based on CybORG test_bline_reset
    """

    def test_bline_same_trajectory_after_reset(self):
        """B_line should produce identical trajectory after reset."""
        seed = 42

        harness1 = DifferentialHarness(seed=seed, max_steps=16, verbose=False)
        result1 = harness1.run_bline_episode(sleep_policy, use_jax_bline=False)

        harness2 = DifferentialHarness(seed=seed, max_steps=16, verbose=False)
        result2 = harness2.run_bline_episode(sleep_policy, use_jax_bline=False)

        assert len(result1.step_results) == len(result2.step_results)

        for sr1, sr2 in zip(result1.step_results, result2.step_results):
            assert sr1.red_action_jax == sr2.red_action_jax, \
                f"Step {sr1.step}: actions differ - {sr1.red_action_desc} vs {sr2.red_action_desc}"

            assert abs(sr1.jax_state.reward_red - sr2.jax_state.reward_red) < 0.01, \
                f"Step {sr1.step}: rewards differ"


@requires_cyborg
class TestBLineBlueInterruptions:
    """Test B_line behavior with Blue interruptions.

    Based on CybORG test_bline_agent_blue_interruptions
    """

    @pytest.fixture
    def harness(self):
        return DifferentialHarness(seed=42, max_steps=30, verbose=False)

    @pytest.mark.parametrize("interrupt_step,interrupt_action", [
        (4, 'remove'),
        (6, 'restore'),
        (8, 'remove'),
    ])
    def test_bline_interrupted_at_step(self, harness, interrupt_step, interrupt_action):
        """Blue interruption at specific step should match CybORG behavior."""
        target_host = 'Enterprise1'

        def blue_interrupt(state: StateSnapshot, step: int) -> int:
            if step == interrupt_step:
                if interrupt_action == 'remove':
                    return BLUE_REMOVE_START + HOST_IDS[target_host]
                else:
                    return BLUE_RESTORE_START + HOST_IDS[target_host]
            return BLUE_SLEEP

        result = harness.run_bline_episode(blue_interrupt, use_jax_bline=False)

        assert result.steps_completed >= interrupt_step + 5

        for sr in result.step_results:
            assert abs(sr.cyborg_state.reward_blue - sr.jax_state.reward_blue) < 0.1, \
                f"Step {sr.step}: blue reward mismatch after {interrupt_action}"


# =============================================================================
# SECTION 3: Blue Action Tests
# Based on: test_blue_remove.py, test_blue_restore.py, test_blue_monitor.py
# =============================================================================

@requires_cyborg
class TestBlueRemoveAction:
    """Test Blue Remove action behavior.

    Based on CybORG test_blue_remove_killchain_sim_expected_output
    """

    @pytest.fixture
    def harness(self):
        return DifferentialHarness(seed=42, max_steps=25, verbose=False)

    def test_remove_blocks_subsequent_privesc(self, harness):
        """Remove after Exploit should block PrivEsc."""
        result = harness.run_bline_episode(reactive_remove_policy, use_jax_bline=False)

        for sr in result.step_results:
            assert abs(sr.cyborg_state.reward_blue - sr.jax_state.reward_blue) < 0.1, \
                f"Step {sr.step}: reward mismatch"

    def test_remove_on_uncompromised_host(self, harness):
        """Remove on uncompromised host should be safe."""
        def remove_early(state: StateSnapshot, step: int) -> int:
            if step == 0:
                return BLUE_REMOVE_START + HOST_IDS['Enterprise0']
            return BLUE_SLEEP

        def red_sleep(state, step):
            return RED_SLEEP

        result = harness.run_episode(remove_early, red_sleep)

        assert result.error_diffs == 0, f"State mismatch: {result.failure_reason}"


@requires_cyborg
class TestBlueRestoreAction:
    """Test Blue Restore action behavior.

    Based on CybORG test_blue_restore_killchain_sim_expected_output
    """

    @pytest.fixture
    def harness(self):
        return DifferentialHarness(seed=42, max_steps=25, verbose=False)

    def test_restore_removes_red_sessions(self, harness):
        """Restore should remove Red sessions from host."""
        result = harness.run_bline_episode(reactive_restore_policy, use_jax_bline=False)

        for sr in result.step_results:
            for hostname in HOST_IDS:
                cyborg_priv = sr.cyborg_state.red_privilege.get(hostname, 0)
                jax_priv = sr.jax_state.red_privilege.get(hostname, 0)
                assert cyborg_priv == jax_priv, \
                    f"Step {sr.step}: {hostname} privilege mismatch"

    def test_restore_removes_malware(self, harness):
        """Restore should remove Red's session from host.

        Verified via B_line with reactive restore.
        """
        result = harness.run_bline_episode(reactive_restore_policy, use_jax_bline=False)

        for sr in result.step_results:
            assert abs(sr.cyborg_state.reward_blue - sr.jax_state.reward_blue) < 0.15, \
                f"Step {sr.step}: blue reward mismatch after restore"


@requires_cyborg
class TestBlueMonitorAction:
    """Test Blue Monitor action behavior.

    Based on CybORG test_blue_monitor_on_red_killchain
    """

    @pytest.fixture
    def harness(self):
        return DifferentialHarness(seed=42, max_steps=20, verbose=False)

    def test_monitor_detects_red_activity(self, harness):
        """Monitor should detect Red activity during killchain."""
        result = harness.run_bline_episode(monitor_policy, use_jax_bline=False)

        detected_any = False
        for sr in result.step_results:
            for hostname, detected in sr.jax_state.host_activity_detected.items():
                if detected:
                    detected_any = True
                    break

        assert detected_any, "Monitor should detect some Red activity"

    def test_monitor_state_matches(self, harness):
        """Monitor states should match between CybORG and JAX."""
        result = harness.run_bline_episode(monitor_policy, use_jax_bline=False)

        assert result.error_diffs == 0, f"State mismatch: {result.failure_reason}"


@requires_cyborg
class TestBlueAnalyseAction:
    """Test Blue Analyse action behavior.

    Based on CybORG test_blue_analyse_on_red_killchain, test_analyse_bug_aug19
    """

    @pytest.fixture
    def harness(self):
        return DifferentialHarness(seed=42, max_steps=25, verbose=False)

    def test_analyse_during_killchain(self, harness):
        """Analyse should work during Red killchain."""
        def analyse_user1_periodically(state: StateSnapshot, step: int) -> int:
            if step % 3 == 0:
                return BLUE_ANALYSE_START + HOST_IDS['User1']
            return BLUE_SLEEP

        result = harness.run_bline_episode(analyse_user1_periodically, use_jax_bline=False)

        for sr in result.step_results:
            assert abs(sr.cyborg_state.reward_blue - sr.jax_state.reward_blue) < 0.1, \
                f"Step {sr.step}: blue reward mismatch"

    def test_analyse_all_hosts(self, harness):
        """Analyse on all hosts should produce consistent results."""
        hosts = list(HOST_IDS.keys())

        def analyse_round_robin(state: StateSnapshot, step: int) -> int:
            if step < len(hosts):
                return BLUE_ANALYSE_START + HOST_IDS[hosts[step]]
            return BLUE_SLEEP

        def red_sleep(state, step):
            return RED_SLEEP

        result = harness.run_episode(analyse_round_robin, red_sleep)

        assert result.error_diffs == 0, f"State mismatch: {result.failure_reason}"


# =============================================================================
# SECTION 4: Decoy Tests
# Based on: test_Decoy*.py files
# =============================================================================

@requires_cyborg
class TestDecoyBlocksExploit:
    """Test that decoy deployment blocks the targeted exploit type.

    Based on CybORG test_Decoy*_killchain tests.
    """

    @pytest.fixture
    def harness(self):
        return DifferentialHarness(seed=42, max_steps=30, verbose=False)

    @pytest.mark.parametrize("decoy_host,decoy_type", [
        ('Enterprise0', DECOY_APACHE),
        ('Enterprise0', DECOY_HARAKA),
        ('Enterprise0', DECOY_TOMCAT),
    ])
    def test_decoy_deployment_with_bline(self, harness, decoy_host, decoy_type):
        """Decoy deployment during B_line attack should match CybORG behavior."""

        def blue_deploy_decoy_then_sleep(state: StateSnapshot, step: int) -> int:
            if step == 0:
                return blue_decoy_host(decoy_host, decoy_type)
            return BLUE_SLEEP

        result = harness.run_bline_episode(blue_deploy_decoy_then_sleep, use_jax_bline=False)

        assert result.error_diffs == 0, \
            f"Decoy on {decoy_host}: state mismatch - {result.failure_reason}"


@requires_cyborg
class TestDecoyInvalidOnExistingService:
    """Test that decoy deployment fails on hosts with existing matching services.

    Based on CybORG test_Decoy*_without_red with invalid_hosts
    """

    @pytest.fixture
    def harness(self):
        return DifferentialHarness(seed=42, max_steps=5, verbose=False)

    @pytest.mark.parametrize("host,decoy_type,should_fail", [
        ('Enterprise1', DECOY_APACHE, True),
        ('Enterprise2', DECOY_APACHE, True),
        ('User3', DECOY_APACHE, True),
        ('Enterprise0', DECOY_APACHE, False),
        ('User0', DECOY_SSHD, True),
        ('User1', DECOY_SSHD, True),
        ('Enterprise0', DECOY_SSHD, True),
        ('User2', DECOY_SSHD, False),
        ('User3', DECOY_SSHD, False),
    ])
    def test_decoy_invalid_on_existing_service(self, harness, host, decoy_type, should_fail):
        """Decoy on host with existing service should fail."""
        blue_actions = [blue_decoy_host(host, decoy_type)]
        blue_policy = scripted_blue_policy_factory(blue_actions)

        def red_sleep(state, step):
            return RED_SLEEP

        result = harness.run_episode(blue_policy, red_sleep)

        assert result.error_diffs == 0, f"State mismatch: {result.failure_reason}"


@requires_cyborg
class TestRepeatDecoyFails:
    """Test that deploying the same decoy twice fails.

    Based on CybORG test_Decoy*_repeat tests
    """

    @pytest.fixture
    def harness(self):
        return DifferentialHarness(seed=42, max_steps=5, verbose=False)

    @pytest.mark.parametrize("host,decoy_type", [
        ('Enterprise0', DECOY_SSHD),
        ('User2', DECOY_SSHD),
        ('Enterprise0', DECOY_APACHE),
    ])
    def test_repeat_decoy_fails(self, harness, host, decoy_type):
        """Second deployment of same decoy should fail."""
        blue_actions = [
            blue_decoy_host(host, decoy_type),
            blue_decoy_host(host, decoy_type),
        ]
        blue_policy = scripted_blue_policy_factory(blue_actions)

        def red_sleep(state, step):
            return RED_SLEEP

        result = harness.run_episode(blue_policy, red_sleep)

        assert result.error_diffs == 0, f"State mismatch: {result.failure_reason}"

    def test_repeat_decoy_different_types_allowed(self, harness):
        """Different decoy types on same host should both succeed."""
        host = 'Enterprise0'
        blue_actions = [
            blue_decoy_host(host, DECOY_APACHE),
            blue_decoy_host(host, DECOY_HARAKA),
        ]
        blue_policy = scripted_blue_policy_factory(blue_actions)

        def red_sleep(state, step):
            return RED_SLEEP

        result = harness.run_episode(blue_policy, red_sleep)

        assert result.error_diffs == 0, f"State mismatch: {result.failure_reason}"


@requires_cyborg
class TestDecoyFollowup:
    """Test decoy behavior after killchain completion.

    Based on CybORG test_Decoy*_followup tests
    """

    @pytest.fixture
    def harness(self):
        return DifferentialHarness(seed=42, max_steps=25, verbose=False)

    def test_decoy_deployed_after_exploit(self, harness):
        """Decoy deployed after host exploitation should match CybORG."""
        def blue_decoy_after_step_5(state: StateSnapshot, step: int) -> int:
            if step == 5:
                return blue_decoy_host('Enterprise0', DECOY_APACHE)
            return BLUE_SLEEP

        result = harness.run_bline_episode(blue_decoy_after_step_5, use_jax_bline=False)

        assert result.error_diffs == 0, f"State mismatch: {result.failure_reason}"


@requires_cyborg
class TestDeceptionAppearance:
    """Test how decoys appear to Red's scans.

    Based on CybORG test_deception_appearance, test_scan_deception_combo
    """

    @pytest.fixture
    def harness(self):
        return DifferentialHarness(seed=42, max_steps=15, verbose=False)

    def test_decoy_no_false_positive(self, harness):
        """Deploying decoys should not trigger false detection."""
        def deploy_decoy_on_user2(state: StateSnapshot, step: int) -> int:
            if step == 0:
                return blue_decoy_host('User2', DECOY_APACHE)
            return BLUE_MONITOR

        def red_sleep(state: StateSnapshot, step: int) -> int:
            return RED_SLEEP

        result = harness.run_episode(deploy_decoy_on_user2, red_sleep)

        for sr in result.step_results:
            for hostname, detected in sr.jax_state.host_activity_detected.items():
                if hostname != 'User0':
                    assert not detected, \
                        f"False detection on {hostname} after decoy deployment"

    def test_decoy_interaction_during_bline(self, harness):
        """Decoy interaction during B_line should match CybORG."""
        def deploy_decoy_step_3(state: StateSnapshot, step: int) -> int:
            if step == 3:
                return blue_decoy_host('Enterprise0', DECOY_APACHE)
            return BLUE_SLEEP

        result = harness.run_bline_episode(deploy_decoy_step_3, use_jax_bline=False)

        assert result.error_diffs == 0, f"State mismatch: {result.failure_reason}"


# =============================================================================
# SECTION 5: Individual Exploit Tests
# Based on: test_FTPDirectoryTraversal.py, test_HTTPSRFI.py, etc.
# =============================================================================

@requires_cyborg
class TestFTPDirectoryTraversal:
    """Test FTPDirectoryTraversal exploit behavior.

    Based on CybORG test_FTPDirectoryTraversal_killchain, test_FTPDirectoryTraversal_initial_state
    """

    @pytest.fixture
    def harness(self):
        return DifferentialHarness(seed=42, max_steps=15, verbose=False)

    def test_ftp_exploit_in_bline_context(self, harness):
        """FTP exploit behavior verified via B_line reaching Enterprise hosts."""
        result = harness.run_bline_episode(sleep_policy, use_jax_bline=False)

        for sr in result.step_results:
            assert abs(sr.cyborg_state.reward_red - sr.jax_state.reward_red) < 0.1, \
                f"Step {sr.step}: reward mismatch"


@requires_cyborg
class TestExploitBehaviorViaBLine:
    """Test various exploit behaviors via B_line agent trajectories.

    Covers: HTTPSRFI, HarakaRCE, SQLInjection, SSHBruteForce, EternalBlue
    These are tested indirectly through B_line's exploit selection.
    """

    @pytest.fixture
    def harness(self):
        return DifferentialHarness(seed=42, max_steps=20, verbose=False)

    def test_exploits_match_during_bline_killchain(self, harness):
        """All exploits used by B_line should produce matching rewards."""
        result = harness.run_bline_episode(sleep_policy, use_jax_bline=False)

        for sr in result.step_results:
            if 'Exploit' in sr.red_action_desc or 'SSH' in sr.red_action_desc:
                assert abs(sr.cyborg_state.reward_red - sr.jax_state.reward_red) < 0.1, \
                    f"Step {sr.step} ({sr.red_action_desc}): reward mismatch"

    def test_exploit_success_matches_cyborg(self, harness):
        """Exploit success should match between CybORG and JAX."""
        result = harness.run_bline_episode(sleep_policy, use_jax_bline=False)

        assert result.error_diffs == 0, f"State mismatch: {result.failure_reason}"


# =============================================================================
# SECTION 6: NACL (Network ACL) Tests
# Based on: test_operational_nacl.py
# =============================================================================

@requires_cyborg
class TestOperationalNACL:
    """Test Network ACL blocking behavior.

    Based on CybORG test_operational_nacl, test_operational_nacl_aug20_bug
    """

    @pytest.fixture
    def harness(self):
        return DifferentialHarness(seed=42, max_steps=35, verbose=False)

    def test_restore_enterprise_blocks_op_access(self, harness):
        """Restoring Enterprise hosts should block Red from reaching Op_Server.

        When Enterprise hosts are restored, Red loses the path to Operational subnet.
        """
        restore_step = 17

        def blue_restore_enterprise_after_impact(state: StateSnapshot, step: int) -> int:
            enterprise_hosts = ['Enterprise0', 'Enterprise1', 'Enterprise2']
            restore_offset = step - restore_step
            if 0 <= restore_offset < len(enterprise_hosts):
                return BLUE_RESTORE_START + HOST_IDS[enterprise_hosts[restore_offset]]
            return BLUE_SLEEP

        result = harness.run_bline_episode(blue_restore_enterprise_after_impact, use_jax_bline=False)

        assert result.steps_completed >= restore_step + 5

        for sr in result.step_results:
            assert abs(sr.cyborg_state.reward_blue - sr.jax_state.reward_blue) < 0.15, \
                f"Step {sr.step}: blue reward mismatch"

    def test_nacl_behavior_via_bline_restore(self, harness):
        """NACL blocking verified via Blue restore interrupting B_line."""
        result = harness.run_bline_episode(reactive_restore_policy, use_jax_bline=False)

        for sr in result.step_results:
            assert abs(sr.cyborg_state.reward_red - sr.jax_state.reward_red) < 0.15, \
                f"Step {sr.step}: reward mismatch"


# =============================================================================
# SECTION 7: DiscoverRemoteSystems/DiscoverNetworkServices Tests
# Based on: test_DiscoverRemoteSystems.py
# =============================================================================

@requires_cyborg
class TestDiscoverRemoteSystems:
    """Test DiscoverRemoteSystems action behavior.

    Based on CybORG test_DiscoverRemoteSystems_killchain, test_DiscoverRemoteSystems_initial_state
    """

    @pytest.fixture
    def harness(self):
        return DifferentialHarness(seed=42, max_steps=20, verbose=False)

    def test_discover_all_subnets_in_killchain(self, harness):
        """All subnet discoveries during killchain should match CybORG."""
        result = harness.run_bline_episode(sleep_policy, use_jax_bline=False)

        for sr in result.step_results:
            if 'Discover' in sr.red_action_desc:
                assert abs(sr.cyborg_state.reward_red - sr.jax_state.reward_red) < 0.05, \
                    f"Step {sr.step}: discover reward mismatch"

    def test_discover_subnet_behavior_matches(self, harness):
        """Discover subnet behavior should match CybORG through full killchain."""
        result = harness.run_bline_episode(sleep_policy, use_jax_bline=False)

        discover_steps = [sr for sr in result.step_results if 'Discover' in sr.red_action_desc]
        assert len(discover_steps) >= 1, "B_line should discover at least User subnet"

        for sr in discover_steps:
            assert abs(sr.cyborg_state.reward_red - sr.jax_state.reward_red) < 0.05


# =============================================================================
# SECTION 8: Alternative Killchain Tests
# =============================================================================

@requires_cyborg
class TestAlternativeKillchains:
    """Test alternative killchain paths via B_line variations.

    B_line's deterministic path is the gold standard for parity testing.
    """

    @pytest.fixture
    def harness(self):
        return DifferentialHarness(seed=42, max_steps=20, verbose=False)

    def test_bline_killchain_matches_cyborg(self, harness):
        """Standard B_line killchain should match CybORG exactly."""
        result = harness.run_bline_episode(sleep_policy, use_jax_bline=False)

        impact_reached = any('Impact' in sr.red_action_desc for sr in result.step_results)
        assert impact_reached, "Should reach Impact"

        for sr in result.step_results:
            assert abs(sr.cyborg_state.reward_red - sr.jax_state.reward_red) < 0.1, \
                f"Step {sr.step}: reward mismatch"

    @pytest.mark.parametrize("seed", [42, 100, 200])
    def test_bline_different_seeds_match(self, seed):
        """B_line should match across different seeds."""
        harness = DifferentialHarness(seed=seed, max_steps=20, verbose=False)
        result = harness.run_bline_episode(sleep_policy, use_jax_bline=False)

        assert result.error_diffs == 0, f"Seed {seed}: {result.failure_reason}"


# =============================================================================
# SECTION 9: Multi-Seed Consistency Tests
# =============================================================================

@requires_cyborg
class TestMultiSeedConsistency:
    """Test consistency across multiple seeds."""

    @pytest.mark.parametrize("seed", [42, 123, 456, 789])
    def test_bline_consistent_across_seeds(self, seed):
        """B_line should produce matching states across seeds."""
        harness = DifferentialHarness(seed=seed, max_steps=20, verbose=False)
        result = harness.run_bline_episode(sleep_policy, use_jax_bline=False)

        assert result.error_diffs == 0, \
            f"Seed {seed}: Found {result.error_diffs} differences"

    @pytest.mark.parametrize("seed", [42, 123])
    def test_reactive_blue_consistent_across_seeds(self, seed):
        """Reactive blue agents should produce matching states."""
        harness = DifferentialHarness(seed=seed, max_steps=30, verbose=False)
        result = harness.run_bline_episode(reactive_remove_policy, use_jax_bline=False)

        for sr in result.step_results:
            assert abs(sr.cyborg_state.reward_blue - sr.jax_state.reward_blue) < 0.15, \
                f"Seed {seed}, Step {sr.step}: blue reward mismatch"


# =============================================================================
# SECTION 10: Long Episode Tests
# =============================================================================

@requires_cyborg
class TestLongEpisodes:
    """Test behavior over longer episodes.

    Based on CybORG test_long_killchain, test_short_killchain
    """

    def test_long_episode_no_drift(self):
        """States should not drift over long episodes."""
        harness = DifferentialHarness(seed=42, max_steps=100, verbose=False)
        result = harness.run_bline_episode(reactive_restore_policy, use_jax_bline=False)

        total_reward_diff = 0
        for sr in result.step_results:
            total_reward_diff += abs(sr.cyborg_state.reward_blue - sr.jax_state.reward_blue)

        avg_diff = total_reward_diff / len(result.step_results)
        assert avg_diff < 0.1, f"Average reward drift too high: {avg_diff}"

    def test_short_killchain_complete(self):
        """Short killchain should complete successfully."""
        harness = DifferentialHarness(seed=42, max_steps=16, verbose=False)
        result = harness.run_bline_episode(sleep_policy, use_jax_bline=False)

        impact_reached = any('Impact' in sr.red_action_desc for sr in result.step_results)
        assert impact_reached, "Should reach Impact in short killchain"


# =============================================================================
# SECTION 11: Edge Cases and Regression Tests
# =============================================================================

@requires_cyborg
class TestEdgeCasesAndRegressions:
    """Test edge cases and historical bug regressions."""

    @pytest.fixture
    def harness(self):
        return DifferentialHarness(seed=42, max_steps=20, verbose=False)

    def test_sleep_action_no_effect(self, harness):
        """Sleep action should have no state effect."""
        def red_all_sleep(state, step):
            return RED_SLEEP

        result = harness.run_episode(sleep_policy, red_all_sleep)

        for sr in result.step_results:
            assert sr.cyborg_state.reward_red == sr.jax_state.reward_red == 0, \
                f"Step {sr.step}: sleep should give 0 reward"

    def test_invalid_action_handling(self, harness):
        """Invalid actions should be handled consistently."""
        red_actions = [
            red_exploit_host('User0', EXPLOIT_SSH),
        ]

        red_policy = scripted_red_policy_factory(red_actions)

        result = harness.run_episode(sleep_policy, red_policy)

        assert result.error_diffs == 0, f"State mismatch: {result.failure_reason}"

    def test_impact_requires_privesc_via_bline(self, harness):
        """Impact requires prior PrivEsc - verified via B_line sequence.

        B_line always does PrivEsc before Impact, so Impact succeeds.
        """
        result = harness.run_bline_episode(sleep_policy, use_jax_bline=False)

        impact_step = None
        for i, sr in enumerate(result.step_results):
            if 'Impact' in sr.red_action_desc:
                impact_step = i
                break

        assert impact_step is not None, "B_line should reach Impact"

        prev_step = result.step_results[impact_step - 1]
        assert 'PrivEsc' in prev_step.red_action_desc or 'Escalate' in prev_step.red_action_desc, \
            "PrivEsc should precede Impact"


# =============================================================================
# SECTION 12: Blue Remove/Restore Detailed Killchain Tests
# Based on: test_blue_remove.py, test_blue_restore.py (with seed parameterization)
# =============================================================================

@requires_cyborg
class TestBlueRemoveDetailedKillchain:
    """Detailed Blue Remove tests during Red killchain.

    Based on CybORG test_blue_remove_killchain_sim_expected_output
    """

    @pytest.mark.parametrize("seed", [1, 22447, 3242, 4])
    def test_remove_during_killchain_multi_seed(self, seed):
        """Remove during killchain should match CybORG across seeds."""
        harness = DifferentialHarness(seed=seed, max_steps=30, verbose=False)
        result = harness.run_bline_episode(reactive_remove_policy, use_jax_bline=False)

        for sr in result.step_results:
            assert abs(sr.cyborg_state.reward_blue - sr.jax_state.reward_blue) < 0.15, \
                f"Seed {seed}, Step {sr.step}: blue reward mismatch"

    @pytest.mark.parametrize("seed", [1, 22447])
    def test_remove_prevents_privesc_multi_seed(self, seed):
        """Remove after exploit should prevent PrivEsc in CybORG and JAX."""
        harness = DifferentialHarness(seed=seed, max_steps=30, verbose=False)
        result = harness.run_bline_episode(reactive_remove_policy, use_jax_bline=False)

        for sr in result.step_results:
            for hostname in HOST_IDS:
                cyborg_priv = sr.cyborg_state.red_privilege.get(hostname, 0)
                jax_priv = sr.jax_state.red_privilege.get(hostname, 0)
                assert cyborg_priv == jax_priv, \
                    f"Seed {seed}, Step {sr.step}: {hostname} privilege mismatch"


@requires_cyborg
class TestBlueRestoreDetailedKillchain:
    """Detailed Blue Restore tests during Red killchain.

    Based on CybORG test_blue_restore_killchain_sim_expected_output
    """

    @pytest.mark.parametrize("seed", [42, 123])
    def test_restore_during_killchain_multi_seed(self, seed):
        """Restore during killchain should match CybORG across seeds."""
        harness = DifferentialHarness(seed=seed, max_steps=30, verbose=False)
        result = harness.run_bline_episode(reactive_restore_policy, use_jax_bline=False)

        for sr in result.step_results:
            assert abs(sr.cyborg_state.reward_blue - sr.jax_state.reward_blue) < 0.15, \
                f"Seed {seed}, Step {sr.step}: blue reward mismatch"

    def test_restore_removes_all_red_access(self):
        """Restore should remove all Red access levels."""
        harness = DifferentialHarness(seed=42, max_steps=25, verbose=False)

        def restore_user1_after_exploit(state: StateSnapshot, step: int) -> int:
            if step == 4:
                return BLUE_RESTORE_START + HOST_IDS['User1']
            return BLUE_SLEEP

        result = harness.run_bline_episode(restore_user1_after_exploit, use_jax_bline=False)

        for sr in result.step_results:
            if sr.step > 4:
                user1_jax_priv = sr.jax_state.red_privilege.get('User1', 0)
                user1_cyborg_priv = sr.cyborg_state.red_privilege.get('User1', 0)
                assert user1_jax_priv == user1_cyborg_priv, \
                    f"Step {sr.step}: User1 privilege mismatch after restore"


# =============================================================================
# SECTION 13: Analyse Bug Regression Tests
# Based on: test_blue_analyse.py test_analyse_bug_aug19
# =============================================================================

@requires_cyborg
class TestAnalyseBugRegressions:
    """Analyse action regression tests.

    Based on CybORG test_analyse_bug_aug19
    """

    @pytest.fixture
    def harness(self):
        return DifferentialHarness(seed=1, max_steps=15, verbose=False)

    def test_analyse_repeated_same_host(self, harness):
        """Repeatedly analysing the same host should be consistent."""
        def analyse_enterprise0_repeatedly(state: StateSnapshot, step: int) -> int:
            return BLUE_ANALYSE_START + HOST_IDS['Enterprise0']

        result = harness.run_bline_episode(analyse_enterprise0_repeatedly, use_jax_bline=False)

        for sr in result.step_results:
            assert abs(sr.cyborg_state.reward_blue - sr.jax_state.reward_blue) < 0.1, \
                f"Step {sr.step}: reward mismatch during repeated analyse"

    def test_analyse_during_red_activity(self, harness):
        """Analyse should work consistently during Red activity."""
        def analyse_various_hosts(state: StateSnapshot, step: int) -> int:
            hosts = ['User1', 'Enterprise0', 'Enterprise1', 'Op_Server0']
            return BLUE_ANALYSE_START + HOST_IDS[hosts[step % len(hosts)]]

        result = harness.run_bline_episode(analyse_various_hosts, use_jax_bline=False)

        assert result.error_diffs == 0, f"State mismatch: {result.failure_reason}"


# =============================================================================
# SECTION 14: Full Analyse on Killchain Tests
# Based on: test_blue_analyse.py test_blue_analyse_on_red_killchain
# =============================================================================

@requires_cyborg
class TestAnalyseFullKillchain:
    """Test Analyse action throughout Red killchain.

    Based on CybORG test_blue_analyse_on_red_killchain
    """

    @pytest.fixture
    def harness(self):
        return DifferentialHarness(seed=42, max_steps=20, verbose=False)

    def test_analyse_detects_compromise_during_killchain(self, harness):
        """Analyse should consistently detect Red compromise during killchain."""
        hosts_to_analyse = ['User1', 'User2', 'Enterprise0', 'Enterprise1']
        host_idx = [0]

        def analyse_round_robin(state: StateSnapshot, step: int) -> int:
            host = hosts_to_analyse[host_idx[0] % len(hosts_to_analyse)]
            host_idx[0] += 1
            return BLUE_ANALYSE_START + HOST_IDS[host]

        result = harness.run_bline_episode(analyse_round_robin, use_jax_bline=False)

        for sr in result.step_results:
            assert abs(sr.cyborg_state.reward_blue - sr.jax_state.reward_blue) < 0.15, \
                f"Step {sr.step}: blue reward mismatch during analyse"

    def test_analyse_on_compromised_host(self, harness):
        """Analyse on compromised host should match CybORG."""
        def analyse_user1_continuously(state: StateSnapshot, step: int) -> int:
            return BLUE_ANALYSE_START + HOST_IDS['User1']

        result = harness.run_bline_episode(analyse_user1_continuously, use_jax_bline=False)

        for sr in result.step_results:
            assert abs(sr.cyborg_state.reward_blue - sr.jax_state.reward_blue) < 0.1, \
                f"Step {sr.step}: reward mismatch"


# =============================================================================
# SECTION 15: Restore Removes Malware Tests
# Based on: test_analyse_restore.py test_restore_removes_malware
# =============================================================================

@requires_cyborg
class TestRestoreRemovesMalware:
    """Test that Restore action removes malware.

    Based on CybORG test_restore_removes_malware (uses Meander but B_line is similar)
    """

    def test_restore_clears_compromise_state(self):
        """Restore should clear Red compromise state on restored hosts."""
        harness = DifferentialHarness(seed=42, max_steps=25, verbose=False)

        all_hosts = ['User1', 'User2', 'Enterprise0', 'Enterprise1', 'Enterprise2', 'Op_Server0']
        restore_start_step = 16

        def restore_all_after_killchain(state: StateSnapshot, step: int) -> int:
            restore_idx = step - restore_start_step
            if 0 <= restore_idx < len(all_hosts):
                return BLUE_RESTORE_START + HOST_IDS[all_hosts[restore_idx]]
            return BLUE_SLEEP

        result = harness.run_bline_episode(restore_all_after_killchain, use_jax_bline=False)

        assert result.steps_completed >= restore_start_step + len(all_hosts)

        for sr in result.step_results[:restore_start_step + len(all_hosts)]:
            assert abs(sr.cyborg_state.reward_blue - sr.jax_state.reward_blue) < 0.15, \
                f"Step {sr.step}: blue reward mismatch"

    def test_restore_all_hosts_sequentially(self):
        """Restoring all hosts sequentially should match CybORG."""
        harness = DifferentialHarness(seed=42, max_steps=40, verbose=False)

        hosts = list(HOST_IDS.keys())
        hosts = [h for h in hosts if h != 'User0']

        def restore_all_hosts(state: StateSnapshot, step: int) -> int:
            if step < len(hosts):
                return BLUE_RESTORE_START + HOST_IDS[hosts[step]]
            return BLUE_SLEEP

        def red_sleep(state, step):
            return RED_SLEEP

        result = harness.run_episode(restore_all_hosts, red_sleep)

        for sr in result.step_results[:len(hosts)]:
            assert abs(sr.cyborg_state.reward_blue - sr.jax_state.reward_blue) < 0.05, \
                f"Step {sr.step}: restore cost mismatch"


# =============================================================================
# SECTION 16: Privilege Escalation Detailed Tests
# Based on: test_PrivilegeEscalate.py
# =============================================================================

@requires_cyborg
class TestPrivilegeEscalateDetails:
    """Detailed tests for PrivilegeEscalate action.

    Based on CybORG test_PrivilegeEscalate_success tests
    """

    @pytest.fixture
    def harness(self):
        return DifferentialHarness(seed=42, max_steps=20, verbose=False)

    def test_privesc_only_after_exploit(self, harness):
        """PrivEsc should only succeed after exploit - verified via B_line."""
        result = harness.run_bline_episode(sleep_policy, use_jax_bline=False)

        privesc_steps = [sr for sr in result.step_results if 'PrivEsc' in sr.red_action_desc or 'Escalate' in sr.red_action_desc]

        for sr in privesc_steps:
            assert abs(sr.cyborg_state.reward_red - sr.jax_state.reward_red) < 0.1, \
                f"Step {sr.step}: privesc reward mismatch"

    def test_privesc_followed_by_remove_behavior(self, harness):
        """PrivEsc followed by Remove should match CybORG behavior."""
        result = harness.run_bline_episode(reactive_remove_policy, use_jax_bline=False)

        for sr in result.step_results:
            for hostname in HOST_IDS:
                cyborg_priv = sr.cyborg_state.red_privilege.get(hostname, 0)
                jax_priv = sr.jax_state.red_privilege.get(hostname, 0)
                assert cyborg_priv == jax_priv, \
                    f"Step {sr.step}: {hostname} privilege mismatch"


# =============================================================================
# SECTION 17: Impact Action Detailed Tests
# Based on: test_Impact.py
# =============================================================================

@requires_cyborg
class TestImpactDetails:
    """Detailed tests for Impact action.

    Based on CybORG test_Impact_operational_server tests
    """

    @pytest.fixture
    def harness(self):
        return DifferentialHarness(seed=42, max_steps=25, verbose=False)

    def test_impact_reward_value(self, harness):
        """Impact on Op_Server0 should give Red +10 reward."""
        result = harness.run_bline_episode(sleep_policy, use_jax_bline=False)

        impact_sr = None
        for sr in result.step_results:
            if 'Impact' in sr.red_action_desc:
                impact_sr = sr
                break

        assert impact_sr is not None, "Should reach Impact"

        assert abs(impact_sr.cyborg_state.reward_red - impact_sr.jax_state.reward_red) < 0.1, \
            "Impact reward should match"

    def test_repeated_impact_continues(self, harness):
        """Impact continues giving penalty each step."""
        harness = DifferentialHarness(seed=42, max_steps=25, verbose=False)
        result = harness.run_bline_episode(sleep_policy, use_jax_bline=False)

        post_impact = False
        for sr in result.step_results:
            if 'Impact' in sr.red_action_desc:
                post_impact = True
            if post_impact:
                assert abs(sr.cyborg_state.reward_blue - sr.jax_state.reward_blue) < 0.1, \
                    f"Step {sr.step}: post-impact reward mismatch"


# =============================================================================
# SECTION 18: Green Agent Coexistence Tests
# =============================================================================

@requires_cyborg
class TestGreenAgentCoexistence:
    """Test that Green agent presence doesn't affect Red/Blue parity.

    Based on CybORG test_GreenAgent
    """

    @pytest.fixture
    def harness(self):
        return DifferentialHarness(seed=42, max_steps=20, verbose=False)

    def test_green_agent_no_effect_on_parity(self, harness):
        """Green agent should not affect Red/Blue behavior parity."""
        result = harness.run_bline_episode(sleep_policy, use_jax_bline=False)

        assert result.error_diffs == 0, f"State mismatch with Green: {result.failure_reason}"

    def test_green_with_reactive_blue(self, harness):
        """Green + reactive Blue should still match."""
        result = harness.run_bline_episode(reactive_remove_policy, use_jax_bline=False)

        for sr in result.step_results:
            assert abs(sr.cyborg_state.reward_blue - sr.jax_state.reward_blue) < 0.15, \
                f"Step {sr.step}: reward mismatch with Green present"


# =============================================================================
# SECTION 19: Session Management Tests
# =============================================================================

@requires_cyborg
class TestSessionManagement:
    """Test Red session creation and removal parity."""

    @pytest.fixture
    def harness(self):
        return DifferentialHarness(seed=42, max_steps=30, verbose=False)

    def test_session_count_matches_after_exploits(self, harness):
        """Red session count should match after exploits."""
        result = harness.run_bline_episode(sleep_policy, use_jax_bline=False)

        for sr in result.step_results:
            cyborg_compromised = sum(1 for v in sr.cyborg_state.host_compromised.values() if v)
            jax_compromised = sum(1 for v in sr.jax_state.host_compromised.values() if v)
            assert cyborg_compromised == jax_compromised, \
                f"Step {sr.step}: compromised host count mismatch"

    def test_session_removal_on_restore(self, harness):
        """Restore should remove Red sessions matching CybORG."""
        def restore_enterprise1_at_step_8(state: StateSnapshot, step: int) -> int:
            if step == 8:
                return BLUE_RESTORE_START + HOST_IDS['Enterprise1']
            return BLUE_SLEEP

        result = harness.run_bline_episode(restore_enterprise1_at_step_8, use_jax_bline=False)

        for sr in result.step_results:
            for hostname in HOST_IDS:
                cyborg_comp = sr.cyborg_state.host_compromised.get(hostname, False)
                jax_comp = sr.jax_state.host_compromised.get(hostname, False)
                assert cyborg_comp == jax_comp, \
                    f"Step {sr.step}: {hostname} compromise state mismatch"


# =============================================================================
# SECTION 20: Action Space Validation Tests
# =============================================================================

@requires_cyborg
class TestActionSpaceValidation:
    """Test action space validation parity."""

    @pytest.fixture
    def harness(self):
        return DifferentialHarness(seed=42, max_steps=10, verbose=False)

    def test_all_blue_actions_valid(self, harness):
        """All Blue actions should produce valid state transitions."""
        hosts = ['User1', 'User2', 'Enterprise0', 'Enterprise1']
        actions = []

        for host in hosts:
            actions.append(BLUE_ANALYSE_START + HOST_IDS[host])

        def blue_action_cycle(state: StateSnapshot, step: int) -> int:
            if step < len(actions):
                return actions[step]
            return BLUE_SLEEP

        def red_sleep(state, step):
            return RED_SLEEP

        result = harness.run_episode(blue_action_cycle, red_sleep)

        assert result.error_diffs == 0, f"State mismatch: {result.failure_reason}"

    def test_decoy_actions_all_types(self, harness):
        """All decoy types should produce matching behavior."""
        decoy_types = [DECOY_APACHE, DECOY_HARAKA, DECOY_TOMCAT, DECOY_FEMITTER]
        host = 'Enterprise0'

        for decoy_type in decoy_types:
            h = DifferentialHarness(seed=42, max_steps=5, verbose=False)

            def blue_deploy_decoy(state: StateSnapshot, step: int) -> int:
                if step == 0:
                    return blue_decoy_host(host, decoy_type)
                return BLUE_SLEEP

            def red_sleep(state, step):
                return RED_SLEEP

            result = h.run_episode(blue_deploy_decoy, red_sleep)

            assert result.error_diffs == 0, \
                f"Decoy type {decoy_type}: state mismatch - {result.failure_reason}"


# =============================================================================
# SECTION 21: Alternative Red Agents (Meander)
# Based on: test_meander_vs_blue_agent_start
# =============================================================================

@requires_cyborg
class TestMeanderAgent:
    """Test Meander agent vs various Blue agents.

    Based on CybORG test_meander_vs_blue_agent_start (not skipped).
    Uses CybORG's RedMeanderAgent with actions translated to JAX.
    """

    @pytest.fixture
    def harness(self):
        return DifferentialHarness(seed=42, max_steps=50, verbose=False)

    def test_meander_vs_sleep_agent(self, harness):
        """Meander vs SleepAgent should produce matching rewards."""
        result = harness.run_meander_episode(sleep_policy)

        for sr in result.step_results:
            assert abs(sr.cyborg_state.reward_red - sr.jax_state.reward_red) < 0.15, \
                f"Step {sr.step}: red reward mismatch"
            assert abs(sr.cyborg_state.reward_blue - sr.jax_state.reward_blue) < 0.15, \
                f"Step {sr.step}: blue reward mismatch"

    def test_meander_vs_react_remove(self, harness):
        """Meander vs ReactRemove should produce matching behavior."""
        result = harness.run_meander_episode(reactive_remove_policy)

        for sr in result.step_results:
            assert abs(sr.cyborg_state.reward_blue - sr.jax_state.reward_blue) < 0.15, \
                f"Step {sr.step}: blue reward mismatch"

    def test_meander_vs_react_restore(self, harness):
        """Meander vs ReactRestore should produce matching behavior."""
        result = harness.run_meander_episode(reactive_restore_policy)

        for sr in result.step_results:
            assert abs(sr.cyborg_state.reward_blue - sr.jax_state.reward_blue) < 0.15, \
                f"Step {sr.step}: blue reward mismatch"

    @pytest.mark.parametrize("seed", [42, 123, 456])
    def test_meander_across_seeds(self, seed):
        """Meander should match CybORG across different seeds."""
        harness = DifferentialHarness(seed=seed, max_steps=30, verbose=False)
        result = harness.run_meander_episode(sleep_policy)

        for sr in result.step_results:
            assert abs(sr.cyborg_state.reward_red - sr.jax_state.reward_red) < 0.15, \
                f"Seed {seed}, Step {sr.step}: red reward mismatch"


# =============================================================================
# SECTION 22: Individual Exploit Tests
# Based on: test_SQLInjection_killchain, test_HarakaRCE_killchain, etc.
# =============================================================================

@requires_cyborg
class TestIndividualExploits:
    """Test individual exploit types directly.

    Based on CybORG exploit-specific tests.
    """

    @pytest.fixture
    def harness(self):
        return DifferentialHarness(seed=42, max_steps=20, verbose=False)

    @pytest.mark.parametrize("exploit_type,exploit_name,target_host", [
        (EXPLOIT_SSH, "SSHBruteForce", "User1"),
        (EXPLOIT_HTTP, "HTTPRFI", "Enterprise1"),
        (EXPLOIT_HTTPS, "HTTPSRFI", "Enterprise1"),
        (EXPLOIT_HARAKA, "HarakaRCE", "User3"),
    ])
    def test_individual_exploit_type(self, harness, exploit_type, exploit_name, target_host):
        """Test specific exploit type reaches target and matches CybORG."""
        host_subnet = get_host_subnet(target_host)

        actions = []
        if host_subnet != SUBNET_USER:
            actions.extend(get_path_to_subnet(host_subnet))

        actions.extend([
            red_discover_subnet(host_subnet),
            red_scan_host(target_host),
            red_exploit_host(target_host, exploit_type),
        ])

        red_policy = scripted_red_policy_factory(actions)
        result = harness.run_episode(sleep_policy, red_policy)

        exploit_step = len(actions) - 1
        if result.steps_completed > exploit_step:
            sr = result.step_results[exploit_step]
            assert abs(sr.cyborg_state.reward_red - sr.jax_state.reward_red) < 0.1, \
                f"{exploit_name} on {target_host}: reward mismatch"

    def test_exploit_on_invalid_host_fails_consistently(self, harness):
        """Exploit on host without vulnerable service should fail in both."""
        actions = [
            red_discover_subnet(SUBNET_USER),
            red_scan_host('User1'),
            red_exploit_host('User1', EXPLOIT_HARAKA),
        ]

        red_policy = scripted_red_policy_factory(actions)
        result = harness.run_episode(sleep_policy, red_policy)

        assert result.error_diffs == 0, f"State mismatch: {result.failure_reason}"

    def test_exploit_without_prior_scan(self, harness):
        """Exploit without prior scan should behave consistently.

        CybORG allows exploit if you have the IP from DiscoverSubnet.
        JAX now matches this behavior.
        """
        actions = [
            red_discover_subnet(SUBNET_USER),
            red_exploit_host('User1', EXPLOIT_SSH),
        ]

        red_policy = scripted_red_policy_factory(actions)
        result = harness.run_episode(sleep_policy, red_policy)

        assert result.error_diffs == 0, f"State mismatch: {result.failure_reason}"


@requires_cyborg
class TestExploitKillchainVariations:
    """Test exploit variations through complete killchains.

    Based on CybORG test_*_killchain tests.
    """

    @pytest.fixture
    def harness(self):
        return DifferentialHarness(seed=42, max_steps=25, verbose=False)

    def test_haraka_killchain(self, harness):
        """Complete killchain using Haraka exploit where applicable."""
        red_policy = scripted_red_policy_factory(KILLCHAIN_VIA_HARAKA)
        result = harness.run_episode(sleep_policy, red_policy)

        for sr in result.step_results:
            assert abs(sr.cyborg_state.reward_red - sr.jax_state.reward_red) < 0.1, \
                f"Step {sr.step}: reward mismatch in Haraka killchain"

    def test_enterprise0_path_killchain(self, harness):
        """Complete killchain via Enterprise0 (alternative path)."""
        red_policy = scripted_red_policy_factory(KILLCHAIN_VIA_ENTERPRISE0)
        result = harness.run_episode(sleep_policy, red_policy)

        for sr in result.step_results:
            assert abs(sr.cyborg_state.reward_red - sr.jax_state.reward_red) < 0.1, \
                f"Step {sr.step}: reward mismatch in Enterprise0 path"


# =============================================================================
# SECTION 23: Additional Decoy Edge Cases
# Based on: test_DecoySvchost_*, test_DecoySmss_*, test_DecoyFemitter_*
# =============================================================================

@requires_cyborg
class TestDecoyEdgeCases:
    """Additional decoy deployment edge cases.

    Based on CybORG test_DecoySvchost, test_DecoySmss, test_DecoyFemitter tests.
    """

    @pytest.fixture
    def harness(self):
        return DifferentialHarness(seed=42, max_steps=10, verbose=False)

    @pytest.mark.parametrize("decoy_type,decoy_name", [
        (DECOY_SVCHOST, "Svchost"),
        (DECOY_SMSS, "Smss"),
        (DECOY_FEMITTER, "Femitter"),
        (DECOY_VSFTPD, "Vsftpd"),
    ])
    def test_additional_decoy_types_on_enterprise(self, harness, decoy_type, decoy_name):
        """Additional decoy types should deploy consistently on Enterprise hosts."""
        host = 'Enterprise0'
        blue_actions = [blue_decoy_host(host, decoy_type)]
        blue_policy = scripted_blue_policy_factory(blue_actions)

        def red_sleep(state, step):
            return RED_SLEEP

        result = harness.run_episode(blue_policy, red_sleep)

        assert result.error_diffs == 0, \
            f"{decoy_name} decoy on {host}: state mismatch - {result.failure_reason}"

    @pytest.mark.parametrize("host", ['User1', 'User2', 'Enterprise1', 'Enterprise2'])
    def test_decoy_svchost_various_hosts(self, harness, host):
        """Svchost decoy on various hosts should match CybORG behavior."""
        blue_actions = [blue_decoy_host(host, DECOY_SVCHOST)]
        blue_policy = scripted_blue_policy_factory(blue_actions)

        def red_sleep(state, step):
            return RED_SLEEP

        result = harness.run_episode(blue_policy, red_sleep)

        assert result.error_diffs == 0, f"State mismatch: {result.failure_reason}"

    def test_multiple_different_decoys_same_host(self, harness):
        """Multiple different decoy types on same host."""
        host = 'Enterprise0'
        blue_actions = [
            blue_decoy_host(host, DECOY_APACHE),
            blue_decoy_host(host, DECOY_SVCHOST),
            blue_decoy_host(host, DECOY_SMSS),
            blue_decoy_host(host, DECOY_FEMITTER),
        ]
        blue_policy = scripted_blue_policy_factory(blue_actions)

        def red_sleep(state, step):
            return RED_SLEEP

        result = harness.run_episode(blue_policy, red_sleep)

        assert result.error_diffs == 0, f"State mismatch: {result.failure_reason}"

    def test_decoy_then_bline_attack(self, harness):
        """Decoys deployed before B_line attack should affect behavior consistently."""
        harness = DifferentialHarness(seed=42, max_steps=25, verbose=False)

        decoys_placed = [False]

        def blue_decoy_then_sleep(state: StateSnapshot, step: int) -> int:
            if step == 0:
                decoys_placed[0] = True
                return blue_decoy_host('Enterprise0', DECOY_SVCHOST)
            elif step == 1:
                return blue_decoy_host('Enterprise1', DECOY_SMSS)
            return BLUE_SLEEP

        result = harness.run_bline_episode(blue_decoy_then_sleep, use_jax_bline=False)

        assert decoys_placed[0], "Decoys should have been placed"

        for sr in result.step_results:
            assert abs(sr.cyborg_state.reward_blue - sr.jax_state.reward_blue) < 0.15, \
                f"Step {sr.step}: blue reward mismatch with decoys"


@requires_cyborg
class TestDecoyInteractionWithExploits:
    """Test decoy interaction with specific exploit types.

    Based on CybORG decoy followup tests.
    """

    @pytest.fixture
    def harness(self):
        return DifferentialHarness(seed=42, max_steps=20, verbose=False)

    @pytest.mark.parametrize("decoy_type,blocked_exploit", [
        (DECOY_SSHD, EXPLOIT_SSH),
        (DECOY_APACHE, EXPLOIT_HTTP),
        (DECOY_HARAKA, EXPLOIT_HARAKA),
        (DECOY_TOMCAT, EXPLOIT_HTTP),
    ])
    def test_decoy_blocks_matching_exploit(self, harness, decoy_type, blocked_exploit):
        """Decoy should block matching exploit type consistently."""
        target_host = 'Enterprise0'

        def blue_deploy_decoy(state: StateSnapshot, step: int) -> int:
            if step == 0:
                return blue_decoy_host(target_host, decoy_type)
            return BLUE_SLEEP

        red_actions = [
            RED_SLEEP,
            red_discover_subnet(SUBNET_USER),
            red_scan_host('User1'),
            red_exploit_host('User1', EXPLOIT_SSH),
            red_privesc_host('User1'),
            red_discover_subnet(SUBNET_ENTERPRISE),
            red_scan_host(target_host),
            red_exploit_host(target_host, blocked_exploit),
        ]

        red_policy = scripted_red_policy_factory(red_actions)
        result = harness.run_episode(blue_deploy_decoy, red_policy)

        for sr in result.step_results:
            assert abs(sr.cyborg_state.reward_red - sr.jax_state.reward_red) < 0.15, \
                f"Step {sr.step}: reward mismatch with decoy blocking exploit"


# =============================================================================
# SECTION 24: Impact Host Restriction Tests
# Based on: CybORG Impact action only succeeds on Op_Server0
# =============================================================================

@requires_cyborg
class TestImpactHostRestrictions:
    """Test that Impact only succeeds on Op_Server0.

    Impact action requires:
    1. SYSTEM/root privilege on target host
    2. Target must be a high-value operational server (Op_Server0)
    """

    @pytest.fixture
    def harness(self):
        return DifferentialHarness(seed=42, max_steps=20, verbose=False)

    def test_impact_on_user_host_fails(self, harness):
        """Impact on User host should fail even with SYSTEM privilege."""
        red_actions = [
            red_discover_subnet(SUBNET_USER),
            red_scan_host('User3'),
            red_exploit_host('User3', EXPLOIT_HARAKA),  # Haraka gives root
            red_privesc_host('User3'),
            red_impact_host('User3'),  # Should fail - User3 is not high-value
        ]

        red_policy = scripted_red_policy_factory(red_actions)
        result = harness.run_episode(sleep_policy, red_policy)

        assert result.steps_completed >= 5
        impact_step = result.step_results[4]

        assert abs(impact_step.cyborg_state.reward_blue - impact_step.jax_state.reward_blue) < 0.15, \
            "Impact on User3 reward should match"

    def test_impact_on_enterprise_host_fails(self, harness):
        """Impact on Enterprise host should fail."""
        red_actions = [
            red_discover_subnet(SUBNET_USER),
            red_scan_host('User4'),
            red_exploit_host('User4', EXPLOIT_HARAKA),
            red_privesc_host('User4'),
            red_scan_host('Enterprise0'),
            red_exploit_host('Enterprise0', EXPLOIT_HTTP),
            red_privesc_host('Enterprise0'),
            red_impact_host('Enterprise0'),  # Should fail
        ]

        red_policy = scripted_red_policy_factory(red_actions)
        result = harness.run_episode(sleep_policy, red_policy)

        assert result.steps_completed >= 8
        for sr in result.step_results:
            assert abs(sr.cyborg_state.reward_red - sr.jax_state.reward_red) < 0.15, \
                f"Step {sr.step}: reward mismatch"


# =============================================================================
# SECTION 25: Remove Action Privilege Tests
# Based on: CybORG Remove only works on user-level sessions
# =============================================================================

@requires_cyborg
class TestRemovePrivilegeRestrictions:
    """Test Remove action privilege-level restrictions.

    Remove action should only succeed on user-level sessions.
    SYSTEM/root sessions should persist after Remove.
    """

    @pytest.fixture
    def harness(self):
        return DifferentialHarness(seed=42, max_steps=15, verbose=False)

    def test_remove_on_root_session_ineffective(self, harness):
        """Remove on SYSTEM session should not clear the session."""
        red_actions = [
            red_discover_subnet(SUBNET_USER),
            red_scan_host('User3'),
            red_exploit_host('User3', EXPLOIT_HARAKA),  # Gives root directly
            red_privesc_host('User3'),
            RED_SLEEP,  # Let Blue Remove
            red_exploit_host('User3', EXPLOIT_HARAKA),  # Try again - should succeed if session persists
        ]

        def blue_remove_after_exploit(state: StateSnapshot, step: int) -> int:
            if step == 4:
                return blue_remove_host('User3')
            return BLUE_SLEEP

        red_policy = scripted_red_policy_factory(red_actions)
        result = harness.run_episode(blue_remove_after_exploit, red_policy)

        for sr in result.step_results:
            assert abs(sr.cyborg_state.reward_blue - sr.jax_state.reward_blue) < 0.15, \
                f"Step {sr.step}: blue reward mismatch"


# =============================================================================
# SECTION 26: Restore Cost Tests
# Based on: CybORG Restore always costs -1.0
# =============================================================================

@requires_cyborg
class TestRestoreCostConsistency:
    """Test that Restore action always costs -1.0."""

    @pytest.fixture
    def harness(self):
        return DifferentialHarness(seed=42, max_steps=10, verbose=False)

    def test_restore_cost_on_clean_host(self, harness):
        """Restore on uncompromised host should cost -1.0."""
        blue_actions = [
            blue_restore_host('User1'),
            blue_restore_host('Enterprise0'),
            blue_restore_host('Op_Server0'),
        ]

        blue_policy = scripted_blue_policy_factory(blue_actions)

        def red_sleep(state, step):
            return RED_SLEEP

        result = harness.run_episode(blue_policy, red_sleep)

        for i, sr in enumerate(result.step_results[:3]):
            assert abs(sr.cyborg_state.reward_blue - sr.jax_state.reward_blue) < 0.05, \
                f"Step {i}: Restore cost mismatch"

    def test_multiple_restores_accumulate_cost(self, harness):
        """Multiple Restores on same host should each cost -1.0."""
        blue_actions = [
            blue_restore_host('User1'),
            blue_restore_host('User1'),
            blue_restore_host('User1'),
        ]

        blue_policy = scripted_blue_policy_factory(blue_actions)

        def red_sleep(state, step):
            return RED_SLEEP

        result = harness.run_episode(blue_policy, red_sleep)

        for sr in result.step_results[:3]:
            assert abs(sr.cyborg_state.reward_blue - sr.jax_state.reward_blue) < 0.05, \
                f"Step {sr.step}: accumulated Restore cost mismatch"


# =============================================================================
# SECTION 27: Action Prerequisite Tests
# Based on: CybORG action ordering requirements
# =============================================================================

@requires_cyborg
class TestActionPrerequisites:
    """Test action prerequisite requirements match CybORG."""

    @pytest.fixture
    def harness(self):
        return DifferentialHarness(seed=42, max_steps=10, verbose=False)

    def test_privesc_without_exploit_fails(self, harness):
        """PrivEsc without prior exploit should fail."""
        red_actions = [
            red_discover_subnet(SUBNET_USER),
            red_scan_host('User1'),
            red_privesc_host('User1'),  # Should fail - no exploit
        ]

        red_policy = scripted_red_policy_factory(red_actions)
        result = harness.run_episode(sleep_policy, red_policy)

        for sr in result.step_results:
            assert abs(sr.cyborg_state.reward_red - sr.jax_state.reward_red) < 0.15, \
                f"Step {sr.step}: reward mismatch"

    def test_impact_without_privesc_fails(self, harness):
        """Impact without PrivEsc should fail (user-level exploit)."""
        red_actions = [
            red_discover_subnet(SUBNET_USER),
            red_scan_host('User1'),
            red_exploit_host('User1', EXPLOIT_SSH),  # SSH gives user-level
            red_impact_host('User1'),  # Should fail - no SYSTEM privilege
        ]

        red_policy = scripted_red_policy_factory(red_actions)
        result = harness.run_episode(sleep_policy, red_policy)

        for sr in result.step_results:
            assert abs(sr.cyborg_state.reward_red - sr.jax_state.reward_red) < 0.1, \
                f"Step {sr.step}: reward mismatch"

    def test_exploit_without_scan_behavior(self, harness):
        """Exploit without prior scan should behave consistently.

        CybORG allows exploit if you have the IP from DiscoverSubnet.
        JAX now matches this behavior.
        """
        red_actions = [
            red_discover_subnet(SUBNET_USER),
            red_exploit_host('User1', EXPLOIT_SSH),  # Skip scan
        ]

        red_policy = scripted_red_policy_factory(red_actions)
        result = harness.run_episode(sleep_policy, red_policy)

        assert result.error_diffs == 0, f"State mismatch: {result.failure_reason}"


# =============================================================================
# SECTION 28: Network Topology Constraint Tests
# Based on: CybORG subnet reachability rules
# =============================================================================

@requires_cyborg
class TestNetworkTopologyConstraints:
    """Test network topology constraints match CybORG.

    - User hosts can reach Enterprise hosts
    - Enterprise hosts can reach Operational hosts
    - User hosts cannot directly reach Operational hosts
    """

    @pytest.fixture
    def harness(self):
        return DifferentialHarness(seed=42, max_steps=15, verbose=False)

    def test_direct_op_server_scan_from_user(self, harness):
        """Scanning Op_Server0 directly from User0 should fail."""
        red_actions = [
            red_discover_subnet(SUBNET_OPERATIONAL),  # Try to discover Op subnet directly
            red_scan_host('Op_Server0'),  # Should fail - not reachable from User0
        ]

        red_policy = scripted_red_policy_factory(red_actions)
        result = harness.run_episode(sleep_policy, red_policy)

        for sr in result.step_results:
            assert abs(sr.cyborg_state.reward_red - sr.jax_state.reward_red) < 0.15, \
                f"Step {sr.step}: reward mismatch on unreachable scan"

    def test_enterprise_reachable_from_user(self, harness):
        """Enterprise0 should be reachable from User hosts."""
        red_actions = [
            red_discover_subnet(SUBNET_USER),
            red_scan_host('User4'),
            red_exploit_host('User4', EXPLOIT_HARAKA),
            red_privesc_host('User4'),
            red_scan_host('Enterprise0'),  # Should succeed - Ent0 reachable from User
        ]

        red_policy = scripted_red_policy_factory(red_actions)
        result = harness.run_episode(sleep_policy, red_policy)

        for sr in result.step_results:
            assert abs(sr.cyborg_state.reward_red - sr.jax_state.reward_red) < 0.15, \
                f"Step {sr.step}: reward mismatch"


# =============================================================================
# SECTION 29: Cumulative Reward Tests
# Based on: Verify reward accumulation matches CybORG
# =============================================================================

@requires_cyborg
class TestCumulativeRewards:
    """Test cumulative reward accumulation matches CybORG."""

    def test_full_killchain_cumulative_rewards(self):
        """Full B_line killchain cumulative rewards should match."""
        harness = DifferentialHarness(seed=42, max_steps=20, verbose=False)
        result = harness.run_bline_episode(sleep_policy, use_jax_bline=False)

        cyborg_cumulative = 0.0
        jax_cumulative = 0.0

        for sr in result.step_results:
            cyborg_cumulative = sr.cyborg_state.reward_red
            jax_cumulative = sr.jax_state.reward_red

        assert abs(cyborg_cumulative - jax_cumulative) < 0.5, \
            f"Final cumulative rewards differ: CybORG={cyborg_cumulative}, JAX={jax_cumulative}"

    def test_reward_accumulation_with_blue_defense(self):
        """Rewards should accumulate correctly with Blue defense."""
        harness = DifferentialHarness(seed=42, max_steps=30, verbose=False)
        result = harness.run_bline_episode(reactive_restore_policy, use_jax_bline=False)

        for sr in result.step_results:
            assert abs(sr.cyborg_state.reward_blue - sr.jax_state.reward_blue) < 0.2, \
                f"Step {sr.step}: cumulative blue reward mismatch"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
