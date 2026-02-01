"""Tests for multi-topology support in differential testing.

These tests verify that the differential testing infrastructure
works with various network topologies, not just Scenario2.
"""

import pytest
import jax
import jax.numpy as jnp

from jaxmarl.environments.cage import CageEnv
from jaxmarl.environments.cage.config import (
    get_scenario,
    ScenarioConfig,
    SCENARIOS,
)
from jaxmarl.environments.cage.actions import (
    BLUE_SLEEP, BLUE_MONITOR,
    get_blue_action_offsets, get_red_action_offsets,
)

from tests.differential.action_translator import (
    describe_jax_blue_action,
    describe_jax_red_action,
    get_host_mappings,
    get_subnet_mappings,
)
from tests.differential.state_comparator import (
    extract_jax_state,
    get_host_ids,
)
from tests.differential.harness import DifferentialHarness, sleep_policy


SCENARIO_HOST_COUNTS = {
    'Scenario2': 13,
    'hosts_2': 25,
    'hosts_3': 37,
}

SCENARIO_NAMES = ['Scenario2', 'hosts_2', 'hosts_3']


@pytest.fixture(scope="module")
def configs():
    """Load all scenario configs once per module."""
    return {name: get_scenario(name) for name in SCENARIO_NAMES}


@pytest.fixture(scope="module")
def envs(configs):
    """Create all CageEnv instances once per module (caches JIT)."""
    result = {}
    for name, cfg in configs.items():
        env = CageEnv(config=cfg, max_steps=100)
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key)
        actions = {'blue': jnp.array(BLUE_SLEEP), 'red': jnp.array(0)}
        env.step_env(key, state, actions)
        result[name] = env
    return result


@pytest.fixture(scope="module")
def scenario2_harness():
    """Shared harness for Scenario2 differential tests."""
    return DifferentialHarness(seed=42, max_steps=10, scenario='Scenario2')


class TestScenarioConfigLoading:
    """Test that scenario configs load with expected host counts."""

    @pytest.mark.parametrize("scenario_name,expected_hosts", [
        ('Scenario2', 13),
        ('hosts_2', 25),
        ('hosts_3', 37),
    ])
    def test_scenario_loads_with_expected_hosts(self, configs, scenario_name, expected_hosts):
        config = configs[scenario_name]
        assert config.num_hosts == expected_hosts
        assert len(config.hosts) == expected_hosts
        assert len(config.host_ids) == expected_hosts

    @pytest.mark.parametrize("scenario_name", SCENARIO_NAMES)
    def test_scenario_has_three_subnets(self, configs, scenario_name):
        config = configs[scenario_name]
        assert config.num_subnets == 3
        assert 'User' in config.subnet_ids
        assert 'Enterprise' in config.subnet_ids
        assert 'Operational' in config.subnet_ids

    @pytest.mark.parametrize("scenario_name", SCENARIO_NAMES)
    def test_scenario_has_agents(self, configs, scenario_name):
        config = configs[scenario_name]
        red_agents = config.get_red_agents()
        blue_agents = config.get_blue_agents()
        assert len(red_agents) >= 1
        assert len(blue_agents) >= 1


class TestJaxEnvWithScenarios:
    """Test JAX environment reset/step with different scenarios."""

    @pytest.mark.parametrize("scenario_name", SCENARIO_NAMES)
    def test_env_reset(self, envs, configs, scenario_name):
        env = envs[scenario_name]
        config = configs[scenario_name]

        key = jax.random.PRNGKey(42)
        obs, state = env.reset(key)

        assert 'blue' in obs
        assert 'red' in obs
        assert state.host_compromised.shape[0] == config.num_hosts

    @pytest.mark.parametrize("scenario_name", SCENARIO_NAMES)
    def test_env_step(self, envs, scenario_name):
        env = envs[scenario_name]

        key = jax.random.PRNGKey(42)
        obs, state = env.reset(key)

        key, subkey = jax.random.split(key)
        actions = {
            'blue': jnp.array(BLUE_SLEEP),
            'red': jnp.array(0),
        }
        obs, state, rewards, dones, info = env.step_env(subkey, state, actions)

        assert 'blue' in rewards
        assert 'red' in rewards

    @pytest.mark.parametrize("scenario_name", SCENARIO_NAMES)
    def test_env_multiple_steps(self, envs, scenario_name):
        env = envs[scenario_name]

        key = jax.random.PRNGKey(42)
        obs, state = env.reset(key)

        for _ in range(10):
            key, subkey = jax.random.split(key)
            actions = {
                'blue': jnp.array(BLUE_MONITOR),
                'red': jnp.array(0),
            }
            obs, state, rewards, dones, info = env.step_env(subkey, state, actions)

        assert int(state.time) == 10


class TestActionSpaceSizing:
    """Test that action space sizes scale with host count."""

    @pytest.mark.parametrize("scenario_name", SCENARIO_NAMES)
    def test_blue_action_space_scales_with_hosts(self, envs, configs, scenario_name):
        env = envs[scenario_name]
        config = configs[scenario_name]

        blue_action_size = env.action_space('blue').n
        assert blue_action_size > config.num_hosts

    @pytest.mark.parametrize("scenario_name", SCENARIO_NAMES)
    def test_red_action_space_scales_with_hosts(self, envs, configs, scenario_name):
        env = envs[scenario_name]
        config = configs[scenario_name]

        red_action_size = env.action_space('red').n
        assert red_action_size > config.num_hosts

    def test_larger_scenario_has_larger_action_space(self, envs):
        env_small = envs['Scenario2']
        env_large = envs['hosts_2']

        assert env_large.action_space('blue').n > env_small.action_space('blue').n
        assert env_large.action_space('red').n > env_small.action_space('red').n


class TestActionDescriptions:
    """Test action descriptions work with different configs."""

    @pytest.mark.parametrize("scenario_name", SCENARIO_NAMES)
    def test_blue_sleep_description(self, configs, scenario_name):
        config = configs[scenario_name]
        desc = describe_jax_blue_action(BLUE_SLEEP, config)
        assert desc == "Sleep"

    @pytest.mark.parametrize("scenario_name", SCENARIO_NAMES)
    def test_blue_monitor_description(self, configs, scenario_name):
        config = configs[scenario_name]
        desc = describe_jax_blue_action(BLUE_MONITOR, config)
        assert desc == "Monitor"

    @pytest.mark.parametrize("scenario_name", SCENARIO_NAMES)
    def test_red_sleep_description(self, configs, scenario_name):
        config = configs[scenario_name]
        desc = describe_jax_red_action(0, config)
        assert desc == "Sleep"

    @pytest.mark.parametrize("scenario_name", SCENARIO_NAMES)
    def test_host_mappings_match_config(self, configs, scenario_name):
        config = configs[scenario_name]
        host_ids, host_names, num_hosts = get_host_mappings(config)

        assert num_hosts == config.num_hosts
        assert host_ids == config.host_ids
        assert host_names == config.host_names


class TestStateExtraction:
    """Test state extraction with different configs."""

    @pytest.mark.parametrize("scenario_name", SCENARIO_NAMES)
    def test_jax_state_extraction(self, envs, configs, scenario_name):
        env = envs[scenario_name]
        config = configs[scenario_name]

        key = jax.random.PRNGKey(42)
        obs, state = env.reset(key)

        snapshot = extract_jax_state(state, env.const, config=config, include_obs=True)

        assert len(snapshot.host_compromised) == config.num_hosts
        assert len(snapshot.red_privilege) == config.num_hosts
        assert all(hostname in config.host_ids for hostname in snapshot.host_compromised.keys())


class TestDifferentialHarnessMultiTopology:
    """Test DifferentialHarness with different scenarios (requires CybORG)."""

    def test_harness_creates_with_scenario(self, scenario2_harness):
        assert scenario2_harness.config.num_hosts == 13

    def test_harness_reset_scenario2(self, scenario2_harness):
        cyborg_state, jax_state = scenario2_harness.reset()

        assert len(jax_state.host_compromised) == 13
        assert 'User0' in jax_state.host_compromised

    def test_harness_step_scenario2(self, scenario2_harness):
        scenario2_harness.reset()

        step_result = scenario2_harness.step(BLUE_SLEEP, 0)

        assert step_result.step == 1
        assert step_result.blue_action_desc == "Sleep"
        assert step_result.red_action_desc == "Sleep"

    def test_harness_episode_scenario2(self):
        harness = DifferentialHarness(
            seed=42,
            max_steps=5,
            scenario='Scenario2',
            check_rewards=True,
        )

        result = harness.run_episode(sleep_policy, lambda s, t: 0)

        assert result.steps_completed == 5
        assert result.passed or result.error_diffs > 0
