"""Tests for CAGE-JAX scalability scenarios and multi-agent support."""

import pytest
import numpy as np
import jax
import jax.numpy as jnp

from jaxmarl.environments.cage import (
    CageEnv, make_cage_env, SCENARIOS,
    create_scenario2_config, create_hosts_2_config, create_hosts_3_config,
    create_hosts_4_config, create_hosts_5_config, create_scalable_config,
    create_multi_agent_config, build_const_from_config,
)
from jaxmarl.environments.cage.actions import (
    compute_blue_action_space_size, compute_red_action_space_size,
)
from jaxmarl.environments.cage.observations import (
    compute_blue_obs_dim, compute_red_obs_dim,
)


class TestScenarioConfigurations:
    """Test scenario configuration generation."""

    def test_scenario2_config(self):
        """Test Scenario2 configuration matches expected values."""
        config = create_scenario2_config()
        assert config.num_hosts == 13
        assert config.num_subnets == 3
        assert len(config.get_red_agents()) == 1
        assert len(config.get_blue_agents()) == 1

    def test_hosts_2_config(self):
        """Test hosts_2 configuration."""
        config = create_hosts_2_config()
        assert config.num_hosts == 25
        assert config.num_subnets == 3

    def test_hosts_3_config(self):
        """Test hosts_3 configuration."""
        config = create_hosts_3_config()
        assert config.num_hosts == 37
        assert config.num_subnets == 3

    def test_hosts_4_config(self):
        """Test hosts_4 configuration."""
        config = create_hosts_4_config()
        assert config.num_hosts == 49
        assert config.num_subnets == 3

    def test_hosts_5_config(self):
        """Test hosts_5 configuration."""
        config = create_hosts_5_config()
        assert config.num_hosts == 62
        assert config.num_subnets == 3

    def test_custom_scalable_config(self):
        """Test custom scalable configuration."""
        config = create_scalable_config(
            num_users=10,
            num_enterprise=5,
            num_op_hosts=5,
            num_op_servers=2,
            name='CustomTest'
        )
        # 10 users + 5 enterprise + 1 defender + 5 op_hosts + 2 op_servers = 23
        assert config.num_hosts == 23
        assert config.name == 'CustomTest'

    def test_multi_agent_config(self):
        """Test multi-agent configuration."""
        base_config = create_scenario2_config()
        config = create_multi_agent_config(
            num_red_agents=3,
            num_blue_agents=2,
            base_config=base_config
        )
        assert config.num_red_agents == 3
        assert config.num_blue_agents == 2
        assert len(config.get_red_agents()) == 3
        assert len(config.get_blue_agents()) == 2


class TestScenarioEnvironments:
    """Test environment creation for all scenarios."""

    @pytest.mark.parametrize("scenario", list(SCENARIOS.keys()))
    def test_env_creation(self, scenario):
        """Test environment can be created for each scenario."""
        env = CageEnv(scenario=scenario)
        info = env.get_scenario_info()
        assert info['num_hosts'] > 0
        assert info['blue_action_space_size'] > 0
        assert info['red_action_space_size'] > 0

    @pytest.mark.parametrize("scenario", list(SCENARIOS.keys()))
    def test_env_reset(self, scenario):
        """Test environment reset for each scenario."""
        env = CageEnv(scenario=scenario)
        key = jax.random.PRNGKey(42)
        obs, state = env.reset(key)

        info = env.get_scenario_info()
        assert obs['blue'].shape[0] == info['blue_obs_dim']
        assert obs['red'].shape[0] == info['red_obs_dim']

    @pytest.mark.parametrize("scenario", list(SCENARIOS.keys()))
    def test_env_step(self, scenario):
        """Test environment step for each scenario."""
        env = CageEnv(scenario=scenario)
        key = jax.random.PRNGKey(42)
        obs, state = env.reset(key)

        actions = {
            'blue': jnp.array(0),
            'red': jnp.array(0),
        }
        key, subkey = jax.random.split(key)
        obs, state, rewards, dones, info = env.step_env(subkey, state, actions)

        assert state.time == 1
        assert 'blue' in rewards
        assert 'red' in rewards

    @pytest.mark.parametrize("scenario", list(SCENARIOS.keys()))
    def test_action_mask(self, scenario):
        """Test action mask for each scenario."""
        env = CageEnv(scenario=scenario)
        key = jax.random.PRNGKey(42)
        obs, state = env.reset(key)

        masks = env.get_avail_actions(state)
        info = env.get_scenario_info()

        assert masks['blue'].shape[0] == info['blue_action_space_size']
        assert masks['red'].shape[0] == info['red_action_space_size']
        assert jnp.any(masks['blue'])
        assert jnp.any(masks['red'])


class TestMultiAgentEnvironments:
    """Test multi-agent functionality."""

    def test_two_red_agents(self):
        """Test environment with 2 red agents."""
        env = CageEnv(scenario='Scenario2', num_red_agents=2)
        assert len(env.red_agents) == 2
        assert 'red_0' in env.agents
        assert 'red_1' in env.agents

    def test_two_blue_agents(self):
        """Test environment with 2 blue agents."""
        env = CageEnv(scenario='Scenario2', num_blue_agents=2)
        assert len(env.blue_agents) == 2
        assert 'blue_0' in env.agents
        assert 'blue_1' in env.agents

    def test_multi_agent_observations(self):
        """Test observations are provided for all agents."""
        env = CageEnv(scenario='Scenario2', num_red_agents=2, num_blue_agents=2)
        key = jax.random.PRNGKey(42)
        obs, state = env.reset(key)

        for agent in env.agents:
            assert agent in obs

    def test_multi_agent_rewards(self):
        """Test rewards are provided for all agents."""
        env = CageEnv(scenario='Scenario2', num_red_agents=2, num_blue_agents=2)
        key = jax.random.PRNGKey(42)
        obs, state = env.reset(key)

        actions = {agent: jnp.array(0) for agent in env.agents}
        key, subkey = jax.random.split(key)
        obs, state, rewards, dones, info = env.step_env(subkey, state, actions)

        for agent in env.agents:
            assert agent in rewards

        # Blue and red rewards should be opposite (zero-sum)
        assert rewards['blue_0'] == rewards['blue_1']
        assert rewards['red_0'] == rewards['red_1']

    def test_multi_agent_with_scalability(self):
        """Test multi-agent with scalability scenario."""
        env = CageEnv(scenario='hosts_2', num_red_agents=3, num_blue_agents=2)
        info = env.get_scenario_info()

        assert info['num_red_agents'] == 3
        assert info['num_blue_agents'] == 2
        assert info['num_hosts'] == 25

        key = jax.random.PRNGKey(42)
        obs, state = env.reset(key)
        assert len(obs) == 5  # 3 red + 2 blue


class TestScalabilityRandomTrajectories:
    """Test random trajectories for scalability scenarios."""

    @pytest.mark.parametrize("scenario", list(SCENARIOS.keys()))
    @pytest.mark.parametrize("seed", range(5))
    def test_random_trajectory(self, scenario, seed):
        """Run random trajectory for each scenario."""
        env = CageEnv(scenario=scenario)
        key = jax.random.PRNGKey(seed)
        obs, state = env.reset(key)

        for step in range(20):
            key, key_blue, key_red, key_step = jax.random.split(key, 4)

            blue_action = jax.random.randint(key_blue, (), 0, env.blue_action_size)
            red_action = jax.random.randint(key_red, (), 0, env.red_action_size)

            actions = {'blue': blue_action, 'red': red_action}
            obs, state, rewards, dones, info = env.step_env(key_step, state, actions)

            assert state.time == step + 1
            assert rewards['red'] >= 0

            if dones['__all__']:
                break


class TestActionSpaceScaling:
    """Test action space scales correctly with host count."""

    def test_blue_action_space_formula(self):
        """Test blue action space formula."""
        for scenario in SCENARIOS.keys():
            config = SCENARIOS[scenario]()
            const = build_const_from_config(config)

            # sleep + monitor + analyse_per_host + remove_per_host + restore_per_host + decoy_per_decoy_host
            expected = 2 + const.num_hosts + const.num_hosts + const.num_hosts + const.num_decoy_hosts * const.num_decoys
            actual = compute_blue_action_space_size(const)

            assert actual == expected, f"{scenario}: expected {expected}, got {actual}"

    def test_red_action_space_formula(self):
        """Test red action space formula."""
        for scenario in SCENARIOS.keys():
            config = SCENARIOS[scenario]()
            const = build_const_from_config(config)

            # 1 sleep + num_subnets discover + num_hosts scan + num_exploits*num_hosts exploit + num_hosts privesc + num_hosts impact
            expected = (1 + const.num_subnets + const.num_hosts +
                       const.num_exploits * const.num_hosts +
                       const.num_hosts + const.num_hosts)
            actual = compute_red_action_space_size(const)

            assert actual == expected, f"{scenario}: expected {expected}, got {actual}"


class TestObservationSpaceScaling:
    """Test observation space scales correctly with host count."""

    def test_blue_obs_formula(self):
        """Test blue observation dimension formula."""
        for scenario in SCENARIOS.keys():
            config = SCENARIOS[scenario]()
            const = build_const_from_config(config)

            expected = const.num_hosts * 4  # 4 features per host
            actual = compute_blue_obs_dim(const)

            assert actual == expected, f"{scenario}: expected {expected}, got {actual}"

    def test_red_obs_formula(self):
        """Test red observation dimension formula."""
        for scenario in SCENARIOS.keys():
            config = SCENARIOS[scenario]()
            const = build_const_from_config(config)

            expected = 1 + const.num_hosts * 3  # 1 success flag + 3 features per host
            actual = compute_red_obs_dim(const)

            assert actual == expected, f"{scenario}: expected {expected}, got {actual}"


class TestNetworkTopology:
    """Test network topology is correctly configured."""

    def test_scenario2_subnet_connectivity(self):
        """Test Scenario2 subnet adjacency matches CybORG NACLs.

        Scenario2 has specific NACL rules:
        - User can reach User and Enterprise, but NOT Operational
        - Enterprise can reach all
        - Operational can reach Enterprise and itself
        """
        from jaxmarl.environments.cage.state import HOST_IDS

        config = SCENARIOS['Scenario2']()
        const = build_const_from_config(config)

        # Get subnet indices dynamically from host assignments
        user_subnet = int(const.host_subnet[HOST_IDS['User0']])
        enterprise_subnet = int(const.host_subnet[HOST_IDS['Enterprise0']])
        operational_subnet = int(const.host_subnet[HOST_IDS['Op_Server0']])

        # User can reach User and Enterprise, but NOT Operational
        assert const.subnet_adjacency[user_subnet, user_subnet]
        assert const.subnet_adjacency[user_subnet, enterprise_subnet]
        assert not const.subnet_adjacency[user_subnet, operational_subnet]

        # Enterprise can reach all
        assert const.subnet_adjacency[enterprise_subnet, user_subnet]
        assert const.subnet_adjacency[enterprise_subnet, enterprise_subnet]
        assert const.subnet_adjacency[enterprise_subnet, operational_subnet]

        # Operational can reach Enterprise and itself
        assert const.subnet_adjacency[operational_subnet, enterprise_subnet]
        assert const.subnet_adjacency[operational_subnet, operational_subnet]

    @pytest.mark.parametrize("scenario", list(SCENARIOS.keys()))
    def test_subnet_adjacency_valid(self, scenario):
        """Test subnet adjacency is a valid boolean matrix."""
        config = SCENARIOS[scenario]()
        const = build_const_from_config(config)

        # Adjacency should be a square boolean matrix
        assert const.subnet_adjacency.shape == (const.num_subnets, const.num_subnets)
        assert const.subnet_adjacency.dtype == jnp.bool_

        # Diagonal should be True (subnets can reach themselves)
        for i in range(const.num_subnets):
            assert const.subnet_adjacency[i, i]

    @pytest.mark.parametrize("scenario", list(SCENARIOS.keys()))
    def test_host_subnet_assignment(self, scenario):
        """Test all hosts are assigned to subnets."""
        config = SCENARIOS[scenario]()
        const = build_const_from_config(config)

        for i in range(const.num_hosts):
            subnet = const.host_subnet[i]
            assert 0 <= subnet < const.num_subnets


class TestRewardScaling:
    """Test rewards scale correctly for different scenarios."""

    @pytest.mark.parametrize("scenario", list(SCENARIOS.keys()))
    def test_max_reward_positive(self, scenario):
        """Test max reward is positive."""
        from jaxmarl.environments.cage.rewards import get_max_red_reward

        config = SCENARIOS[scenario]()
        const = build_const_from_config(config)
        max_reward = get_max_red_reward(const)

        assert max_reward > 0

    @pytest.mark.parametrize("scenario", list(SCENARIOS.keys()))
    def test_rewards_scale_with_hosts(self, scenario):
        """Test that larger scenarios have higher max rewards."""
        from jaxmarl.environments.cage.rewards import get_max_red_reward

        scenario2_config = create_scenario2_config()
        scenario2_const = build_const_from_config(scenario2_config)
        scenario2_max = get_max_red_reward(scenario2_const)

        config = SCENARIOS[scenario]()
        const = build_const_from_config(config)
        max_reward = get_max_red_reward(const)

        if const.num_hosts > scenario2_const.num_hosts:
            assert max_reward >= scenario2_max


class TestMakeFunction:
    """Test the make_cage_env factory function."""

    def test_default_creation(self):
        """Test default environment creation."""
        env = make_cage_env()
        assert env.name == "CAGE-Scenario2"

    def test_scenario_selection(self):
        """Test scenario selection."""
        env = make_cage_env(scenario='hosts_3')
        assert env.name == "CAGE-hosts_3"
        assert env.const.num_hosts == 37

    def test_multi_agent_creation(self):
        """Test multi-agent environment creation."""
        env = make_cage_env(num_red_agents=2, num_blue_agents=3)
        assert env.num_red_agents == 2
        assert env.num_blue_agents == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
