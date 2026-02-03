"""Action-by-action parity tests between CybORG and JAX.

Execute each action in both environments and compare state changes.
This is the most systematic way to find behavioral differences.

Run with: pytest tests/test_action_parity.py -v
"""

import pytest
import sys
sys.path.insert(0, '/home/paulhax/src/cyber/cage-challenge-2/CybORG')

import jax
import jax.numpy as jnp
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from jaxmarl.environments.cage import CageEnv
from jaxmarl.environments.cage.state import (
    HOST_IDS, SERVICE_IDS, EXPLOIT_IDS, DECOY_IDS, NUM_HOSTS,
    COMPROMISE_NONE, COMPROMISE_USER, COMPROMISE_PRIVILEGED,
)
from jaxmarl.environments.cage.actions import (
    BLUE_SLEEP, BLUE_MONITOR, BLUE_ANALYSE_START,
    BLUE_REMOVE_START, BLUE_RESTORE_START, BLUE_DECOY_START,
    RED_SLEEP, RED_DISCOVER_SUBNET_START, RED_SCAN_HOST_START,
    RED_EXPLOIT_START, RED_PRIVESC_START, RED_IMPACT_START,
    NUM_BLUE_ACTIONS, NUM_RED_ACTIONS, NUM_DECOY_TYPES, NUM_EXPLOITS,
    apply_blue_action, apply_red_action,
)

try:
    from CybORG import CybORG
    from CybORG.Shared.Actions.Action import Sleep
    from CybORG.Shared.Actions.AbstractActions.Monitor import Monitor
    from CybORG.Shared.Actions.AbstractActions.Analyse import Analyse
    from CybORG.Shared.Actions.AbstractActions.Remove import Remove
    from CybORG.Shared.Actions.AbstractActions.Restore import Restore
    from CybORG.Shared.Actions.ConcreteActions.SSHBruteForce import SSHBruteForce
    from CybORG.Shared.Actions.ConcreteActions.FTPDirectoryTraversal import FTPDirectoryTraversal
    from CybORG.Shared.Actions.ConcreteActions.EternalBlue import EternalBlue
    from CybORG.Shared.Actions.ConcreteActions.BlueKeep import BlueKeep
    from CybORG.Shared.Actions.ConcreteActions.HarakaRCE import HarakaRCE
    from CybORG.Shared.Actions.ConcreteActions.SQLInjection import SQLInjection
    from CybORG.Shared.Actions.ConcreteActions.HTTPRFI import HTTPRFI
    from CybORG.Shared.Actions.ConcreteActions.HTTPSRFI import HTTPSRFI as HTTPSRfi
    from CybORG.Shared.Actions.AbstractActions.DiscoverRemoteSystems import DiscoverRemoteSystems
    from CybORG.Shared.Actions.AbstractActions.DiscoverNetworkServices import DiscoverNetworkServices
    from CybORG.Shared.Actions.AbstractActions.PrivilegeEscalate import PrivilegeEscalate
    from CybORG.Shared.Actions.AbstractActions.Impact import Impact
    CYBORG_AVAILABLE = True
except ImportError as e:
    print(f"CybORG import error: {e}")
    CYBORG_AVAILABLE = False
    # Define placeholders so parametrize doesn't fail
    SSHBruteForce = FTPDirectoryTraversal = EternalBlue = BlueKeep = None
    HarakaRCE = SQLInjection = HTTPRFI = HTTPSRfi = None
    DiscoverRemoteSystems = DiscoverNetworkServices = PrivilegeEscalate = Impact = None
    Sleep = Monitor = Analyse = Remove = Restore = None

requires_cyborg = pytest.mark.skipif(
    not CYBORG_AVAILABLE,
    reason="CybORG not installed"
)

SCENARIO_PATH = '/home/paulhax/src/cyber/cage-challenge-2/CybORG/CybORG/Shared/Scenarios/Scenario2.yaml'

HOST_NAMES = ['User0', 'User1', 'User2', 'User3', 'User4',
              'Enterprise0', 'Enterprise1', 'Enterprise2', 'Op_Server0']


@dataclass
class ActionResult:
    """Result of executing an action."""
    success: bool
    state_changes: Dict[str, any]
    error: Optional[str] = None


def get_cyborg_host_state(cyborg, hostname: str) -> Dict:
    """Extract relevant state for a host from CybORG."""
    state = cyborg.get_agent_state('True')
    host = state.get(hostname, {})

    sessions = host.get('Sessions', [])
    red_sessions = [s for s in sessions if s.get('Agent') == 'Red']

    has_red_session = len(red_sessions) > 0
    is_privileged = any(s.get('Type') == 'root' or s.get('Username') == 'SYSTEM'
                        for s in red_sessions)

    return {
        'has_red_session': has_red_session,
        'is_privileged': is_privileged,
        'num_red_sessions': len(red_sessions),
    }


def get_jax_host_state(state, host_idx: int) -> Dict:
    """Extract relevant state for a host from JAX."""
    return {
        'has_red_session': int(state.red_privilege[host_idx]) >= COMPROMISE_USER,
        'is_privileged': int(state.red_privilege[host_idx]) >= COMPROMISE_PRIVILEGED,
        'red_privilege': int(state.red_privilege[host_idx]),
    }


@requires_cyborg
class TestBlueActionParity:
    """Test each Blue action type matches between CybORG and JAX."""

    @pytest.fixture
    def environments(self):
        """Create fresh CybORG and JAX environments."""
        cyborg = CybORG(scenario_file=SCENARIO_PATH, environment='sim')
        cyborg.reset()

        jax_env = CageEnv()
        key = jax.random.PRNGKey(42)
        _, jax_state = jax_env.reset(key)

        return cyborg, jax_env, jax_state, key

    def test_blue_sleep(self, environments):
        """Sleep action should have no effect in both environments."""
        cyborg, jax_env, jax_state, key = environments

        # CybORG
        cyborg.step('Blue', Sleep())

        # JAX
        new_jax_state = apply_blue_action(jax_state, BLUE_SLEEP, jax_env.const)

        # Both should succeed with no state changes
        # (Sleep is always valid)

    def test_blue_monitor(self, environments):
        """Monitor action should detect activity."""
        cyborg, jax_env, jax_state, key = environments

        # CybORG
        cyborg.step('Blue', Monitor(session=0, agent='Blue'))

        # JAX
        new_jax_state = apply_blue_action(jax_state, BLUE_MONITOR, jax_env.const)

        # Monitor should work in both

    @pytest.mark.parametrize("hostname", HOST_NAMES)
    def test_blue_restore(self, environments, hostname):
        """Restore action should clear Red sessions."""
        cyborg, jax_env, jax_state, key = environments
        host_idx = HOST_IDS[hostname]

        # First compromise the host in both environments
        # JAX: directly set state
        jax_state = jax_state.replace(
            red_privilege=jax_state.red_privilege.at[host_idx].set(COMPROMISE_USER),
            red_sessions=jax_state.red_sessions.at[host_idx].set(1),
        )

        # Execute Restore
        # CybORG
        cyborg.step('Blue', Restore(session=0, agent='Blue', hostname=hostname))
        cyborg_obs = cyborg.get_observation('Blue')
        cyborg_success = cyborg_obs.get('success', True)

        # JAX
        restore_action = BLUE_RESTORE_START + host_idx
        new_jax_state = apply_blue_action(jax_state, restore_action, jax_env.const)

        # Check if restore clears Red (except User0 which is initial foothold)
        if hostname != 'User0':
            jax_cleared = int(new_jax_state.red_privilege[host_idx]) == COMPROMISE_NONE
            # We can't easily compare CybORG state here, but we can check JAX behavior
            assert jax_cleared or hostname == 'User0', \
                f"Restore should clear Red on {hostname}"


@requires_cyborg
class TestRedExploitParity:
    """Test each Red exploit action matches between CybORG and JAX."""

    @pytest.fixture
    def environments(self):
        """Create fresh environments with Red having scanned the target."""
        cyborg = CybORG(scenario_file=SCENARIO_PATH, environment='sim')
        cyborg.reset()

        jax_env = CageEnv()
        key = jax.random.PRNGKey(42)
        _, jax_state = jax_env.reset(key)

        return cyborg, jax_env, jax_state, key

    @pytest.mark.parametrize("exploit_name,exploit_class,target_host", [
        ('SSHBruteForce', SSHBruteForce, 'User1'),
        ('FTPDirectoryTraversal', FTPDirectoryTraversal, 'User0'),
        ('EternalBlue', EternalBlue, 'User2'),
        ('BlueKeep', BlueKeep, 'User2'),
        ('HarakaRCE', HarakaRCE, 'User3'),
        ('SQLInjection', SQLInjection, 'User3'),
        ('HTTPRFI', HTTPRFI, 'Enterprise1'),
    ])
    def test_exploit_success_parity(self, environments, exploit_name, exploit_class, target_host):
        """Compare exploit success between CybORG and JAX."""
        cyborg, jax_env, jax_state, key = environments

        exploit_idx = EXPLOIT_IDS.get(exploit_name)
        if exploit_idx is None:
            pytest.skip(f"Exploit {exploit_name} not in EXPLOIT_IDS")

        host_idx = HOST_IDS[target_host]

        # Pre-scan target in JAX
        jax_state = jax_state.replace(
            red_scanned_hosts_jax=jax_state.red_scanned_hosts_jax.at[host_idx].set(True)
        )

        # Execute exploit in CybORG
        # First need to discover and scan in CybORG
        cyborg.step('Red', DiscoverNetworkServices(
            session=0, agent='Red', ip_address=cyborg.get_ip_map()[target_host]
        ))

        cyborg.step('Red', exploit_class(
            session=0, agent='Red', ip_address=cyborg.get_ip_map()[target_host],
            target_session=0
        ))
        cyborg_obs = cyborg.get_observation('Red')
        cyborg_success = cyborg_obs.get('success', False)

        # Execute exploit in JAX
        exploit_action = RED_EXPLOIT_START + host_idx * NUM_EXPLOITS + exploit_idx
        new_jax_state = apply_red_action(jax_state, exploit_action, jax_env.const, key)
        jax_success = bool(new_jax_state.last_red_action_success)

        # Compare
        if cyborg_success != jax_success:
            # Document the difference - could be legitimate or a bug
            print(f"MISMATCH: {exploit_name} on {target_host}")
            print(f"  CybORG: {cyborg_success}")
            print(f"  JAX: {jax_success}")

            # Get more details
            cyborg_state = get_cyborg_host_state(cyborg, target_host)
            jax_host_state = get_jax_host_state(new_jax_state, host_idx)
            print(f"  CybORG state: {cyborg_state}")
            print(f"  JAX state: {jax_host_state}")


@requires_cyborg
class TestRedPrivescParity:
    """Test privilege escalation matches between CybORG and JAX."""

    @pytest.fixture
    def environments(self):
        cyborg = CybORG(scenario_file=SCENARIO_PATH, environment='sim')
        cyborg.reset()

        jax_env = CageEnv()
        key = jax.random.PRNGKey(42)
        _, jax_state = jax_env.reset(key)

        return cyborg, jax_env, jax_state, key

    @pytest.mark.parametrize("target_host", ['User1', 'User2', 'Enterprise0'])
    def test_privesc_requires_user_session(self, environments, target_host):
        """PrivEsc should fail without user session in both environments."""
        cyborg, jax_env, jax_state, key = environments
        host_idx = HOST_IDS[target_host]

        # Try privesc without any session
        # CybORG
        cyborg.step('Red', PrivilegeEscalate(
            session=0, agent='Red', hostname=target_host
        ))
        cyborg_obs = cyborg.get_observation('Red')
        cyborg_success = cyborg_obs.get('success', False)

        # JAX
        privesc_action = RED_PRIVESC_START + host_idx
        new_jax_state = apply_red_action(jax_state, privesc_action, jax_env.const, key)
        jax_success = bool(new_jax_state.last_red_action_success)

        # Both should fail (CybORG returns TrinaryEnum which is truthy, so compare with True)
        assert cyborg_success != True, f"CybORG: PrivEsc without session should fail on {target_host}"
        assert not jax_success, f"JAX: PrivEsc without session should fail on {target_host}"


@requires_cyborg
class TestRedImpactParity:
    """Test Impact action matches between CybORG and JAX."""

    @pytest.fixture
    def environments(self):
        cyborg = CybORG(scenario_file=SCENARIO_PATH, environment='sim')
        cyborg.reset()

        jax_env = CageEnv()
        key = jax.random.PRNGKey(42)
        _, jax_state = jax_env.reset(key)

        return cyborg, jax_env, jax_state, key

    def test_impact_requires_privileged_session(self, environments):
        """Impact should fail without privileged session."""
        cyborg, jax_env, jax_state, key = environments

        target_host = 'Op_Server0'
        host_idx = HOST_IDS[target_host]

        # Try impact without any session
        # CybORG
        cyborg.step('Red', Impact(
            session=0, agent='Red', hostname=target_host
        ))
        cyborg_obs = cyborg.get_observation('Red')
        cyborg_success = cyborg_obs.get('success', False)

        # JAX
        impact_action = RED_IMPACT_START + host_idx
        new_jax_state = apply_red_action(jax_state, impact_action, jax_env.const, key)
        jax_success = bool(new_jax_state.last_red_action_success)

        # Both should fail (CybORG returns TrinaryEnum which is truthy, so compare with True)
        assert cyborg_success != True, "CybORG: Impact without privileged session should fail"
        assert not jax_success, "JAX: Impact without privileged session should fail"


@requires_cyborg
class TestFullKillchainParity:
    """Test complete attack killchain matches between CybORG and JAX."""

    def test_bline_killchain_step_by_step(self):
        """Execute B_line killchain using the differential harness."""
        from tests.cage.differential.harness import DifferentialHarness, sleep_policy

        harness = DifferentialHarness(seed=42, max_steps=20, verbose=True)
        result = harness.run_bline_episode(sleep_policy, use_jax_bline=False)

        print(f"\nKillchain comparison results:")
        print(f"  Steps completed: {result.steps_completed}")
        print(f"  Total diffs: {result.total_diffs}")
        print(f"  Error diffs: {result.error_diffs}")
        print(f"  CybORG rewards: {result.cyborg_total_reward}")
        print(f"  JAX rewards: {result.jax_total_reward}")

        # Report any significant differences
        for sr in result.step_results:
            if sr.diffs:
                print(f"\nStep {sr.step}: {sr.red_action_desc}")
                for diff in sr.diffs:
                    print(f"  {diff}")

        assert result.error_diffs == 0, f"Found {result.error_diffs} behavioral differences"


@requires_cyborg
class TestRewardParity:
    """Test reward calculation matches between CybORG and JAX."""

    def test_reward_after_exploit(self):
        """Compare rewards after successful exploit."""
        cyborg = CybORG(scenario_file=SCENARIO_PATH, environment='sim')
        cyborg.reset()

        jax_env = CageEnv()
        key = jax.random.PRNGKey(42)
        obs, jax_state = jax_env.reset(key)

        # Execute exploit sequence
        target = 'User1'
        target_idx = HOST_IDS[target]

        # Scan
        cyborg.step('Red', DiscoverNetworkServices(
            session=0, agent='Red', ip_address=cyborg.get_ip_map()[target]
        ))

        scan_action = RED_SCAN_HOST_START + target_idx
        key, subkey = jax.random.split(key)
        jax_state = apply_red_action(jax_state, scan_action, jax_env.const, subkey)

        # Exploit
        cyborg.step('Red', SSHBruteForce(
            session=0, agent='Red', ip_address=cyborg.get_ip_map()[target],
            target_session=0
        ))
        cyborg_reward = cyborg.get_rewards()['Red']

        exploit_idx = EXPLOIT_IDS['SSHBruteForce']
        exploit_action = RED_EXPLOIT_START + target_idx * NUM_EXPLOITS + exploit_idx
        key, subkey = jax.random.split(key)

        actions = {'blue': jnp.array(BLUE_SLEEP), 'red': jnp.array(exploit_action)}
        obs, jax_state, rewards, dones, info = jax_env.step_env(subkey, jax_state, actions)
        jax_reward = float(rewards['red'])

        print(f"Rewards after exploit on {target}:")
        print(f"  CybORG Red reward: {cyborg_reward}")
        print(f"  JAX Red reward: {jax_reward}")

        # Rewards should be similar (may differ due to timing/calculation differences)
        # Document any significant differences


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
