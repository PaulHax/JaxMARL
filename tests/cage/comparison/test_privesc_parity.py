"""Differential tests comparing CybORG PrivilegeEscalate behavior to JAX.

Every test runs both CybORG and JAX with identical actions and compares results.
These tests verify PrivEsc prerequisites, success/failure behavior, and reward parity.
"""

import pytest
from tests.cage.differential.harness import DifferentialHarness, is_cyborg_available
from tests.cage.differential.state_comparator import COMPROMISE_PRIVILEGED
from tests.cage.comparison.policies import (
    scripted_blue_policy_factory,
    scripted_red_policy_factory,
)
from tests.cage.comparison.scenarios import (
    red_discover_subnet,
    red_scan_host,
    red_exploit_host,
    red_privesc_host,
    blue_remove_host,
    blue_restore_host,
    SUBNET_USER,
    EXPLOIT_SSH,
)
from jaxmarl.environments.cage.actions import BLUE_MONITOR


requires_cyborg = pytest.mark.skipif(
    not is_cyborg_available(),
    reason="CybORG not installed"
)


@requires_cyborg
class TestPrivEscCybORGParity:
    """Differential tests comparing CybORG PrivilegeEscalate behavior to JAX.

    Key CybORG behavior:
    - PrivEsc requires USER-level access on the host
    - Successful PrivEsc elevates to PRIVILEGED (root/SYSTEM) access
    - PrivEsc on host without USER access fails
    - PrivEsc reward is additional confidentiality penalty
    """

    def test_privesc_after_exploit(self):
        """CybORG vs JAX: Standard PrivEsc after exploit succeeds."""
        harness = DifferentialHarness(seed=42, max_steps=10, verbose=False)

        red_actions = [
            red_discover_subnet(SUBNET_USER),
            red_scan_host('User1'),
            red_exploit_host('User1', EXPLOIT_SSH),
            red_privesc_host('User1'),
            0,
        ]

        blue_actions = [BLUE_MONITOR] * 5

        red_policy = scripted_red_policy_factory(red_actions)
        blue_policy = scripted_blue_policy_factory(blue_actions)

        result = harness.run_episode(blue_policy, red_policy)

        privesc_step = result.step_results[3]

        assert abs(privesc_step.cyborg_state.reward_blue - privesc_step.jax_state.reward_blue) < 0.02, \
            f"Blue reward mismatch at PrivEsc: CybORG={privesc_step.cyborg_state.reward_blue}, JAX={privesc_step.jax_state.reward_blue}"

        assert privesc_step.jax_state.red_privilege.get('User1', 0) == COMPROMISE_PRIVILEGED, \
            "Red should have PRIVILEGED access after PrivEsc"

    def test_privesc_without_exploit_fails(self):
        """CybORG vs JAX: PrivEsc without prior exploit should fail."""
        harness = DifferentialHarness(seed=42, max_steps=10, verbose=False)

        red_actions = [
            red_discover_subnet(SUBNET_USER),
            red_scan_host('User1'),
            red_privesc_host('User1'),
            0,
        ]

        blue_actions = [BLUE_MONITOR] * 4

        red_policy = scripted_red_policy_factory(red_actions)
        blue_policy = scripted_blue_policy_factory(blue_actions)

        result = harness.run_episode(blue_policy, red_policy)

        privesc_step = result.step_results[2]

        assert abs(privesc_step.cyborg_state.reward_blue - privesc_step.jax_state.reward_blue) < 0.02, \
            f"Blue reward mismatch: CybORG={privesc_step.cyborg_state.reward_blue}, JAX={privesc_step.jax_state.reward_blue}"

        assert privesc_step.jax_state.red_privilege.get('User1', 0) < COMPROMISE_PRIVILEGED, \
            "Red should NOT have privileged access without prior exploit"

    def test_privesc_on_clean_host_fails(self):
        """CybORG vs JAX: PrivEsc on non-compromised host fails."""
        harness = DifferentialHarness(seed=42, max_steps=5, verbose=False)

        red_actions = [
            red_privesc_host('User1'),
            0,
        ]

        blue_actions = [BLUE_MONITOR] * 2

        red_policy = scripted_red_policy_factory(red_actions)
        blue_policy = scripted_blue_policy_factory(blue_actions)

        result = harness.run_episode(blue_policy, red_policy)

        privesc_step = result.step_results[0]

        assert abs(privesc_step.cyborg_state.reward_blue - privesc_step.jax_state.reward_blue) < 0.02, \
            f"Blue reward mismatch: CybORG={privesc_step.cyborg_state.reward_blue}, JAX={privesc_step.jax_state.reward_blue}"

    @pytest.mark.parametrize("host", ['User1', 'User2', 'User3', 'User4'])
    def test_privesc_on_different_user_hosts(self, host):
        """CybORG vs JAX: PrivEsc behavior should match across User hosts."""
        harness = DifferentialHarness(seed=42, max_steps=8, verbose=False)

        red_actions = [
            red_discover_subnet(SUBNET_USER),
            red_scan_host(host),
            red_exploit_host(host, EXPLOIT_SSH),
            red_privesc_host(host),
            0,
        ]

        blue_actions = [BLUE_MONITOR] * 5

        red_policy = scripted_red_policy_factory(red_actions)
        blue_policy = scripted_blue_policy_factory(blue_actions)

        result = harness.run_episode(blue_policy, red_policy)

        privesc_step = result.step_results[3]
        assert abs(privesc_step.cyborg_state.reward_blue - privesc_step.jax_state.reward_blue) < 0.02, \
            f"PrivEsc on {host}: Blue reward mismatch CybORG={privesc_step.cyborg_state.reward_blue}, JAX={privesc_step.jax_state.reward_blue}"

    def test_privesc_twice_on_same_host(self):
        """CybORG vs JAX: Second PrivEsc on already-privileged host."""
        harness = DifferentialHarness(seed=42, max_steps=10, verbose=False)

        red_actions = [
            red_discover_subnet(SUBNET_USER),
            red_scan_host('User1'),
            red_exploit_host('User1', EXPLOIT_SSH),
            red_privesc_host('User1'),
            red_privesc_host('User1'),
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

    def test_privesc_on_multiple_hosts(self):
        """CybORG vs JAX: PrivEsc on multiple hosts in sequence."""
        harness = DifferentialHarness(seed=42, max_steps=15, verbose=False)

        red_actions = [
            red_discover_subnet(SUBNET_USER),
            red_scan_host('User1'),
            red_exploit_host('User1', EXPLOIT_SSH),
            red_privesc_host('User1'),
            red_scan_host('User2'),
            red_exploit_host('User2', EXPLOIT_SSH),
            red_privesc_host('User2'),
            0,
        ]

        blue_actions = [BLUE_MONITOR] * 8

        red_policy = scripted_red_policy_factory(red_actions)
        blue_policy = scripted_blue_policy_factory(blue_actions)

        result = harness.run_episode(blue_policy, red_policy)

        for sr in result.step_results:
            cyborg_blue = sr.cyborg_state.reward_blue
            jax_blue = sr.jax_state.reward_blue
            assert abs(cyborg_blue - jax_blue) < 0.02, \
                f"Step {sr.step}: Blue reward mismatch CybORG={cyborg_blue}, JAX={jax_blue}"

    def test_privesc_after_remove_fails(self):
        """CybORG vs JAX: PrivEsc after Blue Remove should fail."""
        harness = DifferentialHarness(seed=42, max_steps=12, verbose=False)

        red_actions = [
            red_discover_subnet(SUBNET_USER),
            red_scan_host('User1'),
            red_exploit_host('User1', EXPLOIT_SSH),
            0,  # Sleep while Blue removes
            red_privesc_host('User1'),
            0,
        ]

        blue_actions = [
            BLUE_MONITOR,
            BLUE_MONITOR,
            BLUE_MONITOR,
            blue_remove_host('User1'),
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

    def test_privesc_after_restore_fails(self):
        """CybORG vs JAX: PrivEsc after Blue Restore should fail."""
        harness = DifferentialHarness(seed=42, max_steps=12, verbose=False)

        red_actions = [
            red_discover_subnet(SUBNET_USER),
            red_scan_host('User1'),
            red_exploit_host('User1', EXPLOIT_SSH),
            0,  # Sleep while Blue restores
            red_privesc_host('User1'),
            0,
        ]

        blue_actions = [
            BLUE_MONITOR,
            BLUE_MONITOR,
            BLUE_MONITOR,
            blue_restore_host('User1'),
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

    def test_privesc_on_enterprise_host(self):
        """CybORG vs JAX: PrivEsc on Enterprise host after lateral movement.

        Uses Enterprise1 which is reachable from User subnet (as used in B_line).
        """
        harness = DifferentialHarness(seed=42, max_steps=15, verbose=False)

        red_actions = [
            red_discover_subnet(SUBNET_USER),
            red_scan_host('User1'),
            red_exploit_host('User1', EXPLOIT_SSH),
            red_privesc_host('User1'),
            red_scan_host('Enterprise1'),
            red_exploit_host('Enterprise1', EXPLOIT_SSH),
            red_privesc_host('Enterprise1'),
            0,
        ]

        blue_actions = [BLUE_MONITOR] * 8

        red_policy = scripted_red_policy_factory(red_actions)
        blue_policy = scripted_blue_policy_factory(blue_actions)

        result = harness.run_episode(blue_policy, red_policy)

        for sr in result.step_results:
            cyborg_blue = sr.cyborg_state.reward_blue
            jax_blue = sr.jax_state.reward_blue
            assert abs(cyborg_blue - jax_blue) < 0.02, \
                f"Step {sr.step}: Blue reward mismatch CybORG={cyborg_blue}, JAX={jax_blue}"

    def test_privesc_reward_values(self):
        """CybORG vs JAX: PrivEsc should add confidentiality penalty."""
        harness = DifferentialHarness(seed=42, max_steps=10, verbose=False)

        red_actions = [
            red_discover_subnet(SUBNET_USER),
            red_scan_host('User1'),
            red_exploit_host('User1', EXPLOIT_SSH),
            red_privesc_host('User1'),
            0,
        ]

        blue_actions = [BLUE_MONITOR] * 5

        red_policy = scripted_red_policy_factory(red_actions)
        blue_policy = scripted_blue_policy_factory(blue_actions)

        result = harness.run_episode(blue_policy, red_policy)

        for sr in result.step_results:
            assert abs(sr.cyborg_state.reward_blue - sr.jax_state.reward_blue) < 0.02, \
                f"Step {sr.step}: Blue reward mismatch CybORG={sr.cyborg_state.reward_blue}, JAX={sr.jax_state.reward_blue}"
            assert abs(sr.cyborg_state.reward_red - sr.jax_state.reward_red) < 0.02, \
                f"Step {sr.step}: Red reward mismatch CybORG={sr.cyborg_state.reward_red}, JAX={sr.jax_state.reward_red}"
