"""Differential tests comparing CybORG decoy behavior to JAX.

Every test runs both CybORG and JAX with identical actions and compares results.
Tests cover OS restrictions, port conflicts, and exploit blocking.
"""

import pytest
from tests.cage.differential.harness import DifferentialHarness, is_cyborg_available
from tests.cage.differential.state_comparator import StateSnapshot
from tests.cage.comparison.policies import (
    scripted_blue_policy_factory,
    scripted_red_policy_factory,
    sleep_policy,
)
from tests.cage.comparison.scenarios import (
    red_discover_subnet,
    red_scan_host,
    red_exploit_host,
    red_privesc_host,
    blue_decoy_host,
    SUBNET_USER,
    SUBNET_ENTERPRISE,
    EXPLOIT_SSH,
    EXPLOIT_FTP,
    EXPLOIT_HTTP,
    EXPLOIT_HTTPS,
    EXPLOIT_HARAKA,
    EXPLOIT_ETERNAL,
    EXPLOIT_BLUEKEEP,
    DECOY_APACHE,
    DECOY_FEMITTER,
    DECOY_HARAKA,
    DECOY_SMSS,
    DECOY_SSHD,
    DECOY_SVCHOST,
    DECOY_TOMCAT,
    DECOY_VSFTPD,
)
from jaxmarl.environments.cage.actions import BLUE_SLEEP, BLUE_MONITOR


requires_cyborg = pytest.mark.skipif(
    not is_cyborg_available(),
    reason="CybORG not installed"
)


DECOY_NAMES = {
    DECOY_APACHE: 'DecoyApache',
    DECOY_FEMITTER: 'DecoyFemitter',
    DECOY_HARAKA: 'DecoyHarakaSMPT',
    DECOY_SMSS: 'DecoySmss',
    DECOY_SSHD: 'DecoySSHD',
    DECOY_SVCHOST: 'DecoySvchost',
    DECOY_TOMCAT: 'DecoyTomcat',
    DECOY_VSFTPD: 'DecoyVsftpd',
}

LINUX_ONLY_DECOYS = [DECOY_VSFTPD, DECOY_HARAKA]
WINDOWS_ONLY_DECOYS = [DECOY_SMSS, DECOY_SVCHOST, DECOY_FEMITTER]
ANY_OS_DECOYS = [DECOY_APACHE, DECOY_SSHD, DECOY_TOMCAT]

LINUX_HOSTS = ['User1', 'User2', 'User3', 'User4', 'Defender', 'Op_Host0', 'Op_Host1']
WINDOWS_HOSTS = ['Enterprise0', 'Enterprise1', 'Enterprise2', 'Op_Host2', 'Op_Server0']


@requires_cyborg
class TestDecoyOSRestrictionsCybORGParity:
    """Compare decoy OS restrictions: CybORG vs JAX."""

    @pytest.mark.parametrize("decoy_type,host", [
        (DECOY_VSFTPD, 'Enterprise0'),  # Linux decoy on Windows host - should fail
        (DECOY_HARAKA, 'Enterprise0'),  # Linux decoy on Windows host - should fail
    ])
    def test_linux_decoy_fails_on_windows_host(self, decoy_type, host):
        """CybORG vs JAX: Linux-only decoy on Windows host should fail."""
        harness = DifferentialHarness(seed=42, max_steps=5, verbose=False)

        blue_actions = [
            blue_decoy_host(host, decoy_type),
            BLUE_SLEEP,
        ]

        blue_policy = scripted_blue_policy_factory(blue_actions)
        red_policy = scripted_red_policy_factory([0, 0])

        result = harness.run_episode(blue_policy, red_policy)

        error_diffs = [d for sr in result.step_results
                      for d in sr.diffs if d.severity == 'error']
        assert len(error_diffs) == 0, f"OS restriction mismatch: {error_diffs}"

    @pytest.mark.parametrize("decoy_type,host", [
        (DECOY_SMSS, 'User1'),      # Windows decoy on Linux host - should fail
        (DECOY_SVCHOST, 'User1'),   # Windows decoy on Linux host - should fail
        (DECOY_FEMITTER, 'User1'),  # Windows decoy on Linux host - should fail
    ])
    def test_windows_decoy_fails_on_linux_host(self, decoy_type, host):
        """CybORG vs JAX: Windows-only decoy on Linux host should fail."""
        harness = DifferentialHarness(seed=42, max_steps=5, verbose=False)

        blue_actions = [
            blue_decoy_host(host, decoy_type),
            BLUE_SLEEP,
        ]

        blue_policy = scripted_blue_policy_factory(blue_actions)
        red_policy = scripted_red_policy_factory([0, 0])

        result = harness.run_episode(blue_policy, red_policy)

        error_diffs = [d for sr in result.step_results
                      for d in sr.diffs if d.severity == 'error']
        assert len(error_diffs) == 0, f"OS restriction mismatch: {error_diffs}"

    @pytest.mark.parametrize("decoy_type,host", [
        (DECOY_APACHE, 'User1'),     # Any-OS decoy on Linux
        (DECOY_APACHE, 'Enterprise0'),  # Any-OS decoy on Windows
        (DECOY_SSHD, 'Defender'),    # SSH decoy on Linux
        (DECOY_TOMCAT, 'Enterprise1'),  # Tomcat decoy on Windows
    ])
    def test_anyos_decoy_works_on_both(self, decoy_type, host):
        """CybORG vs JAX: Any-OS decoy should work on both Linux and Windows."""
        harness = DifferentialHarness(seed=42, max_steps=5, verbose=False)

        blue_actions = [
            blue_decoy_host(host, decoy_type),
            BLUE_SLEEP,
        ]

        blue_policy = scripted_blue_policy_factory(blue_actions)
        red_policy = scripted_red_policy_factory([0, 0])

        result = harness.run_episode(blue_policy, red_policy)

        error_diffs = [d for sr in result.step_results
                      for d in sr.diffs if d.severity == 'error']
        assert len(error_diffs) == 0, f"Any-OS decoy failed: {error_diffs}"


@requires_cyborg
class TestDecoyPortConflictsCybORGParity:
    """Compare decoy port conflicts: CybORG vs JAX."""

    @pytest.mark.parametrize("decoy_type,host", [
        (DECOY_SSHD, 'User0'),      # SSH decoy when SSH already running
        (DECOY_SSHD, 'User1'),      # User hosts have SSH service
        (DECOY_SSHD, 'Enterprise0'), # Enterprise hosts have SSH
    ])
    def test_ssh_decoy_port_conflict(self, decoy_type, host):
        """CybORG vs JAX: SSH decoy should fail when SSH port in use."""
        harness = DifferentialHarness(seed=42, max_steps=5, verbose=False)

        blue_actions = [
            blue_decoy_host(host, decoy_type),
            BLUE_SLEEP,
        ]

        blue_policy = scripted_blue_policy_factory(blue_actions)
        red_policy = scripted_red_policy_factory([0, 0])

        result = harness.run_episode(blue_policy, red_policy)

        error_diffs = [d for sr in result.step_results
                      for d in sr.diffs if d.severity == 'error']
        assert len(error_diffs) == 0, f"Port conflict mismatch: {error_diffs}"


@requires_cyborg
class TestDecoyRepeatDeploymentCybORGParity:
    """Compare repeat decoy deployment: CybORG vs JAX."""

    @pytest.mark.parametrize("decoy_type,host", [
        (DECOY_APACHE, 'Defender'),   # Deploy twice
        (DECOY_TOMCAT, 'Defender'),   # Deploy twice
    ])
    def test_repeat_decoy_deployment_fails(self, decoy_type, host):
        """CybORG vs JAX: Second deployment of same decoy should fail."""
        harness = DifferentialHarness(seed=42, max_steps=5, verbose=False)

        blue_actions = [
            blue_decoy_host(host, decoy_type),  # First deployment
            blue_decoy_host(host, decoy_type),  # Second deployment (should fail)
            BLUE_SLEEP,
        ]

        blue_policy = scripted_blue_policy_factory(blue_actions)
        red_policy = scripted_red_policy_factory([0, 0, 0])

        result = harness.run_episode(blue_policy, red_policy)

        error_diffs = [d for sr in result.step_results
                      for d in sr.diffs if d.severity == 'error']
        assert len(error_diffs) == 0, f"Repeat deployment mismatch: {error_diffs}"


@requires_cyborg
class TestDecoyBlocksExploitCybORGParity:
    """Compare decoy exploit blocking: CybORG vs JAX.

    Note: For decoy blocking to work, we need:
    1. Decoy deployed successfully (no port conflict)
    2. Exploit would otherwise succeed (service + preconditions met)

    Enterprise0 is ideal for HTTP decoy testing:
    - Has RFI vulnerability (HTTPRFI would succeed)
    - Does NOT have HTTP service (decoy can be deployed on port 80)
    """

    def test_apache_decoy_blocks_http_exploit_on_enterprise0(self):
        """CybORG vs JAX: Apache decoy should block HTTPRFI on Enterprise0.

        Enterprise0 has RFI vulnerability but no HTTP service running.
        Blue deploys Apache decoy first, then Red attempts HTTP exploit.
        The decoy should block the exploit.
        """
        harness = DifferentialHarness(seed=42, max_steps=15, verbose=False)

        red_actions = [
            0,  # Sleep while Blue deploys decoy
            red_discover_subnet(SUBNET_USER),
            red_scan_host('User1'),
            red_exploit_host('User1', EXPLOIT_SSH),
            red_privesc_host('User1'),
            red_scan_host('Enterprise0'),
            red_exploit_host('Enterprise0', EXPLOIT_HTTP),  # Should be blocked by decoy
            0,  # Sleep
        ]

        blue_actions = [
            blue_decoy_host('Enterprise0', DECOY_APACHE),  # Deploy HTTP decoy
        ] + [BLUE_MONITOR] * (len(red_actions) - 1)

        red_policy = scripted_red_policy_factory(red_actions)
        blue_policy = scripted_blue_policy_factory(blue_actions)

        result = harness.run_episode(blue_policy, red_policy)

        exploit_step = result.step_results[6]

        cyborg_compromised = exploit_step.cyborg_state.host_compromised.get('Enterprise0', 0)
        jax_compromised = exploit_step.jax_state.host_compromised.get('Enterprise0', 0)

        assert cyborg_compromised == 0, f"CybORG should NOT compromise Enterprise0 (decoy blocks), got {cyborg_compromised}"
        assert jax_compromised == 0, f"JAX should NOT compromise Enterprise0 (decoy blocks), got {jax_compromised}"
        assert cyborg_compromised == jax_compromised, \
            f"Decoy blocking mismatch: CybORG={cyborg_compromised}, JAX={jax_compromised}"

    def test_tomcat_decoy_blocks_http_exploit_on_enterprise0(self):
        """CybORG vs JAX: Tomcat decoy should also block HTTPRFI.

        Both DecoyApache and DecoyTomcat block HTTP exploits (port 80).
        """
        harness = DifferentialHarness(seed=42, max_steps=15, verbose=False)

        red_actions = [
            0,  # Sleep while Blue deploys decoy
            red_discover_subnet(SUBNET_USER),
            red_scan_host('User1'),
            red_exploit_host('User1', EXPLOIT_SSH),
            red_privesc_host('User1'),
            red_scan_host('Enterprise0'),
            red_exploit_host('Enterprise0', EXPLOIT_HTTP),  # Should be blocked by decoy
            0,  # Sleep
        ]

        blue_actions = [
            blue_decoy_host('Enterprise0', DECOY_TOMCAT),  # Deploy Tomcat decoy
        ] + [BLUE_MONITOR] * (len(red_actions) - 1)

        red_policy = scripted_red_policy_factory(red_actions)
        blue_policy = scripted_blue_policy_factory(blue_actions)

        result = harness.run_episode(blue_policy, red_policy)

        exploit_step = result.step_results[6]

        cyborg_compromised = exploit_step.cyborg_state.host_compromised.get('Enterprise0', 0)
        jax_compromised = exploit_step.jax_state.host_compromised.get('Enterprise0', 0)

        assert cyborg_compromised == 0, f"CybORG should NOT compromise Enterprise0 (Tomcat decoy blocks), got {cyborg_compromised}"
        assert jax_compromised == 0, f"JAX should NOT compromise Enterprise0 (Tomcat decoy blocks), got {jax_compromised}"
        assert cyborg_compromised == jax_compromised, \
            f"Decoy blocking mismatch: CybORG={cyborg_compromised}, JAX={jax_compromised}"

    @pytest.mark.skip(reason="All bruteforceable hosts in Scenario2 have SSH service - decoy can't be deployed without port conflict")
    def test_ssh_decoy_blocks_ssh_exploit(self):
        """CybORG vs JAX: SSH decoy should block SSHBruteForce.

        Skipped: CybORG's Scenario2 has no hosts that satisfy both:
        1. Have bruteforceable users (required for SSHBruteForce to work)
        2. Don't have SSH service running (required for SSH decoy deployment)

        This is a limitation of the Scenario2 topology, not a bug.
        The HTTP/HTTPS decoy blocking tests (test_apache_decoy_blocks_http_exploit_on_enterprise0,
        test_tomcat_decoy_blocks_http_exploit_on_enterprise0) demonstrate the decoy
        blocking mechanism works correctly.
        """
        pass


@requires_cyborg
class TestDecoyRewardParity:
    """Compare decoy rewards: CybORG vs JAX."""

    def test_decoy_has_no_action_cost(self):
        """CybORG vs JAX: Deploying decoy should not affect Blue reward."""
        harness = DifferentialHarness(seed=42, max_steps=5, verbose=False)

        blue_actions = [
            blue_decoy_host('Defender', DECOY_APACHE),
            BLUE_SLEEP,
        ]

        blue_policy = scripted_blue_policy_factory(blue_actions)
        red_policy = scripted_red_policy_factory([0, 0])

        result = harness.run_episode(blue_policy, red_policy)

        decoy_step = result.step_results[0]
        assert abs(decoy_step.cyborg_state.reward_blue - decoy_step.jax_state.reward_blue) < 0.02, \
            f"Decoy reward mismatch: CybORG={decoy_step.cyborg_state.reward_blue}, JAX={decoy_step.jax_state.reward_blue}"

    def test_failed_decoy_has_no_penalty(self):
        """CybORG vs JAX: Failed decoy deployment should not penalize."""
        harness = DifferentialHarness(seed=42, max_steps=5, verbose=False)

        blue_actions = [
            blue_decoy_host('User1', DECOY_SMSS),  # Windows-only decoy on Linux host
            BLUE_SLEEP,
        ]

        blue_policy = scripted_blue_policy_factory(blue_actions)
        red_policy = scripted_red_policy_factory([0, 0])

        result = harness.run_episode(blue_policy, red_policy)

        decoy_step = result.step_results[0]
        assert abs(decoy_step.cyborg_state.reward_blue - decoy_step.jax_state.reward_blue) < 0.02, \
            f"Failed decoy reward mismatch: CybORG={decoy_step.cyborg_state.reward_blue}, JAX={decoy_step.jax_state.reward_blue}"
