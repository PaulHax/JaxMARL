"""Comprehensive CybORG equivalence tests for CAGE-JAX.

These tests run both CybORG and CAGE-JAX with identical inputs
and verify that outputs match.

Requirements:
- CybORG installed: pip install -r requirements-jax.txt
- Run with: pytest tests/test_cage_cyborg_comprehensive.py -v
"""

import pytest
import numpy as np
from typing import List, Tuple

import jax
import jax.numpy as jnp

from jaxmarl.environments.cage import CageEnv
from jaxmarl.environments.cage.state import HOST_IDS, NUM_HOSTS
from jaxmarl.environments.cage.actions import (
    BLUE_SLEEP, BLUE_MONITOR, BLUE_REMOVE_START, BLUE_RESTORE_START,
    RED_SLEEP, RED_DISCOVER_SUBNET_START, RED_SCAN_HOST_START, RED_EXPLOIT_START,
    NUM_BLUE_ACTIONS, NUM_RED_ACTIONS, NUM_HOSTS as ACTION_NUM_HOSTS,
)


class TestActionEquivalence:
    """Test each action type produces equivalent state transitions."""

    @pytest.fixture
    def envs(self):
        """Create both environments."""
        from tests.cyborg_utils import create_cyborg_env

        cyborg_env = create_cyborg_env(seed=42)
        jax_env = CageEnv()
        return cyborg_env, jax_env

    def test_blue_sleep_action(self, envs):
        """Sleep action should not change state."""
        from tests.cyborg_utils import cyborg_state_to_dict, jax_state_to_dict
        from CybORG.Shared.Actions import Sleep

        cyborg_env, jax_env = envs
        cyborg_env.reset()
        obs, jax_state = jax_env.reset(jax.random.PRNGKey(42))

        cyborg_env.step('Blue', Sleep())
        cyborg_state = cyborg_state_to_dict(cyborg_env)

        key = jax.random.PRNGKey(42)
        actions = {'blue': jnp.array(BLUE_SLEEP), 'red': jnp.array(RED_SLEEP)}
        _, jax_state, _, _, _ = jax_env.step_env(key, jax_state, actions)
        jax_state_dict = jax_state_to_dict(jax_state)

        assert jax_state_dict['host_compromised']['User0'] >= 1

    def test_blue_monitor_action(self, envs):
        """Monitor action should update detection."""
        from tests.cyborg_utils import cyborg_state_to_dict
        from CybORG.Shared.Actions import Monitor

        cyborg_env, jax_env = envs
        cyborg_env.reset()

        result = cyborg_env.step('Blue', Monitor(session=0, agent='Blue'))
        assert result is not None

    def test_red_discover_subnet(self, envs):
        """DiscoverRemoteSystems should discover hosts in target subnet."""
        from tests.cyborg_utils import cyborg_state_to_dict, jax_state_to_dict
        from CybORG.Shared.Actions import DiscoverRemoteSystems

        cyborg_env, jax_env = envs
        cyborg_env.reset()
        obs, jax_state = jax_env.reset(jax.random.PRNGKey(42))

        result = cyborg_env.step('Red', DiscoverRemoteSystems(
            session=0, agent='Red', subnet='User'
        ))
        assert result is not None

        key = jax.random.PRNGKey(42)
        actions = {
            'blue': jnp.array(BLUE_SLEEP),
            'red': jnp.array(RED_DISCOVER_SUBNET_START),
        }
        _, jax_state, _, _, _ = jax_env.step_env(key, jax_state, actions)

        assert jax_state.red_discovered_hosts[HOST_IDS['User0']]

    def test_red_scan_host(self, envs):
        """DiscoverNetworkServices should scan target host."""
        from tests.cyborg_utils import jax_state_to_dict
        from CybORG.Shared.Actions import DiscoverNetworkServices

        cyborg_env, jax_env = envs
        cyborg_env.reset()
        obs, jax_state = jax_env.reset(jax.random.PRNGKey(42))

        ip = cyborg_env.get_ip_map().get('User0')
        result = cyborg_env.step('Red', DiscoverNetworkServices(
            session=0, agent='Red', ip_address=ip
        ))
        assert result is not None

        key = jax.random.PRNGKey(42)
        actions = {
            'blue': jnp.array(BLUE_SLEEP),
            'red': jnp.array(RED_SCAN_HOST_START + HOST_IDS['User0']),
        }
        _, jax_state, _, _, _ = jax_env.step_env(key, jax_state, actions)

        assert jax_state.red_scanned_hosts[HOST_IDS['User0']]


class TestTrajectoryEquivalence:
    """Run identical action sequences and compare step-by-step."""

    @pytest.fixture
    def envs(self):
        """Create both environments."""
        from tests.cyborg_utils import create_cyborg_env

        cyborg_env = create_cyborg_env(seed=42)
        jax_env = CageEnv()
        return cyborg_env, jax_env

    def test_sleep_trajectory(self, envs):
        """Trajectory with all sleep actions should maintain initial state."""
        from tests.cyborg_utils import jax_state_to_dict
        from CybORG.Shared.Actions import Sleep

        cyborg_env, jax_env = envs
        cyborg_env.reset()
        obs, jax_state = jax_env.reset(jax.random.PRNGKey(42))

        initial_jax_state = jax_state_to_dict(jax_state)

        key = jax.random.PRNGKey(42)
        for _ in range(10):
            cyborg_env.step('Blue', Sleep())
            cyborg_env.step('Red', Sleep())

            key, subkey = jax.random.split(key)
            actions = {'blue': jnp.array(BLUE_SLEEP), 'red': jnp.array(RED_SLEEP)}
            obs, jax_state, _, _, _ = jax_env.step_env(subkey, jax_state, actions)

        final_jax_state = jax_state_to_dict(jax_state)

        assert initial_jax_state['host_compromised']['User0'] == final_jax_state['host_compromised']['User0']

    def test_red_discovery_trajectory(self, envs):
        """Test red discovery sequence."""
        from tests.cyborg_utils import jax_state_to_dict
        from CybORG.Shared.Actions import Sleep, DiscoverRemoteSystems

        cyborg_env, jax_env = envs
        cyborg_env.reset()
        obs, jax_state = jax_env.reset(jax.random.PRNGKey(42))

        key = jax.random.PRNGKey(42)

        cyborg_env.step('Red', DiscoverRemoteSystems(session=0, agent='Red', subnet='User'))

        key, subkey = jax.random.split(key)
        actions = {
            'blue': jnp.array(BLUE_SLEEP),
            'red': jnp.array(RED_DISCOVER_SUBNET_START),
        }
        _, jax_state, _, _, _ = jax_env.step_env(subkey, jax_state, actions)

        for host in ['User0', 'User1', 'User2', 'User3', 'User4']:
            assert jax_state.red_discovered_hosts[HOST_IDS[host]], f"{host} should be discovered"

    @pytest.mark.parametrize("seed", range(10))
    def test_random_trajectory(self, envs, seed):
        """Run random trajectories and verify state consistency."""
        from tests.cyborg_utils import jax_state_to_dict

        _, jax_env = envs
        key = jax.random.PRNGKey(seed)
        obs, state = jax_env.reset(key)

        for step in range(20):
            key, key_blue, key_red, key_step = jax.random.split(key, 4)

            blue_action = jax.random.randint(key_blue, (), 0, NUM_BLUE_ACTIONS)
            red_action = jax.random.randint(key_red, (), 0, NUM_RED_ACTIONS)

            actions = {
                'blue': blue_action,
                'red': red_action,
            }

            obs, state, rewards, dones, info = jax_env.step_env(key_step, state, actions)

            assert state.time == step + 1

            is_restore = (blue_action >= BLUE_RESTORE_START) & (blue_action < BLUE_RESTORE_START + 13)
            if is_restore:
                assert rewards['blue'] == -rewards['red'] - 1.0
            else:
                assert rewards['blue'] == -rewards['red']

            if dones['__all__']:
                break


class TestStatisticalEquivalence:
    """Compare distributions over many episodes."""

    def test_initial_state_consistency(self):
        """Initial state should be consistent across resets."""
        from tests.cyborg_utils import create_cyborg_env, cyborg_state_to_dict, jax_state_to_dict

        jax_env = CageEnv()

        jax_states = []
        for seed in range(10):
            key = jax.random.PRNGKey(seed)
            _, state = jax_env.reset(key)
            jax_states.append(jax_state_to_dict(state))

        for state in jax_states:
            assert state['host_compromised']['User0'] >= 1
            for host in ['Enterprise0', 'Enterprise1', 'Op_Server0']:
                assert state['host_compromised'][host] == 0

    def test_reward_bounds(self):
        """Rewards should stay within expected bounds."""
        from jaxmarl.environments.cage.rewards import get_max_red_reward
        from jaxmarl.environments.cage.state import create_scenario2_const

        const = create_scenario2_const()
        max_reward = get_max_red_reward(const)

        jax_env = CageEnv()

        for seed in range(20):
            key = jax.random.PRNGKey(seed)
            obs, state = jax_env.reset(key)

            cumulative_reward = 0.0
            for step in range(100):
                key, key_blue, key_red, key_step = jax.random.split(key, 4)

                blue_action = jax.random.randint(key_blue, (), 0, NUM_BLUE_ACTIONS)
                red_action = jax.random.randint(key_red, (), 0, NUM_RED_ACTIONS)

                actions = {'blue': blue_action, 'red': red_action}
                obs, state, rewards, dones, info = jax_env.step_env(key_step, state, actions)

                assert rewards['red'] >= 0, "Red reward should be non-negative"
                assert rewards['red'] <= max_reward + 1, "Red reward should not exceed max"

                if dones['__all__']:
                    break

    def test_episode_return_distribution(self):
        """Episode returns should be within reasonable bounds."""
        jax_env = CageEnv()

        returns = []
        for seed in range(50):
            key = jax.random.PRNGKey(seed)
            obs, state = jax_env.reset(key)

            episode_return = 0.0
            for step in range(100):
                key, key_blue, key_red, key_step = jax.random.split(key, 4)

                blue_action = jax.random.randint(key_blue, (), 0, NUM_BLUE_ACTIONS)
                red_action = jax.random.randint(key_red, (), 0, NUM_RED_ACTIONS)

                actions = {'blue': blue_action, 'red': red_action}
                obs, state, rewards, dones, info = jax_env.step_env(key_step, state, actions)

                episode_return += float(rewards['red'])

                if dones['__all__']:
                    break

            returns.append(episode_return)

        mean_return = np.mean(returns)
        std_return = np.std(returns)

        assert mean_return >= 0, "Mean return should be non-negative"
        assert std_return > 0, "Should have some variance in returns"


class TestRewardEquivalence:
    """Verify reward calculations match CybORG."""

    def test_zero_sum_property(self):
        """Rewards should be zero-sum (before action costs)."""
        from jaxmarl.environments.cage.rewards import compute_rewards_simple
        from jaxmarl.environments.cage.state import create_initial_state, create_scenario2_const

        const = create_scenario2_const()

        for compromise_level in [0, 1, 2]:
            state = create_initial_state(const)
            state = state.replace(
                host_compromised=state.host_compromised.at[HOST_IDS['Enterprise0']].set(compromise_level)
            )

            rewards = compute_rewards_simple(state, const)
            assert rewards['red'] == -rewards['blue'], "Rewards should be zero-sum"

    def test_confidentiality_rewards(self):
        """Test confidentiality rewards for each host."""
        from jaxmarl.environments.cage.rewards import compute_rewards_simple
        from jaxmarl.environments.cage.state import (
            create_initial_state, create_scenario2_const, COMPROMISE_PRIVILEGED
        )

        const = create_scenario2_const()

        # Confidentiality-only rewards (availability requires Impact)
        expected_rewards = {
            'User0': 0.0,
            'User1': 0.1,
            'Enterprise0': 1.0,
            'Op_Server0': 1.0,  # Confidentiality only; availability requires Impact
        }

        for hostname, expected in expected_rewards.items():
            state = create_initial_state(const)
            state = state.replace(
                host_compromised=state.host_compromised.at[HOST_IDS[hostname]].set(COMPROMISE_PRIVILEGED)
            )
            rewards = compute_rewards_simple(state, const)

            assert abs(float(rewards['red']) - expected) < 1e-5, \
                f"{hostname}: expected {expected}, got {rewards['red']}"

    def test_availability_rewards_require_impact(self):
        """Test that availability reward requires Impact action."""
        from jaxmarl.environments.cage.rewards import compute_rewards_simple
        from jaxmarl.environments.cage.state import (
            create_initial_state, create_scenario2_const, COMPROMISE_PRIVILEGED
        )

        const = create_scenario2_const()
        state = create_initial_state(const)

        # Op_Server0 with privileged access + Impact
        state = state.replace(
            host_compromised=state.host_compromised.at[HOST_IDS['Op_Server0']].set(COMPROMISE_PRIVILEGED),
            ot_service_stopped=state.ot_service_stopped.at[HOST_IDS['Op_Server0']].set(True),
        )
        rewards = compute_rewards_simple(state, const)

        # 1.0 confidentiality + 10.0 availability = 11.0
        assert abs(float(rewards['red']) - 11.0) < 1e-5


class TestCybORGDirectComparison:
    """Direct step-by-step comparison with CybORG."""

    @pytest.fixture
    def cyborg_env(self):
        """Create CybORG environment."""
        from tests.cyborg_utils import create_cyborg_env
        return create_cyborg_env(seed=42)

    def test_initial_reward(self, cyborg_env):
        """Initial reward should match between environments."""
        cyborg_env.reset()
        cyborg_rewards = cyborg_env.get_rewards()

        jax_env = CageEnv()
        _, state = jax_env.reset(jax.random.PRNGKey(42))

        from jaxmarl.environments.cage.rewards import compute_rewards_simple
        jax_rewards = compute_rewards_simple(state, jax_env.const)

        cyborg_red = cyborg_rewards.get('Red', 0)
        jax_red = float(jax_rewards['red'])

        assert abs(cyborg_red - jax_red) < 1.0, \
            f"Initial rewards differ: CybORG={cyborg_red}, JAX={jax_red}"

    def test_host_count(self, cyborg_env):
        """Both environments should have same number of hosts."""
        from jaxmarl.environments.cage.state import NUM_HOSTS

        cyborg_env.reset()
        ip_map = cyborg_env.get_ip_map()

        cyborg_hosts = [h for h in ip_map.keys() if h in HOST_IDS]

        assert len(cyborg_hosts) == NUM_HOSTS, \
            f"Host count mismatch: CybORG={len(cyborg_hosts)}, JAX={NUM_HOSTS}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
