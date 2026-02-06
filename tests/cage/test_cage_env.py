"""Tests for CAGE-JAX environment integration."""

import jax
import jax.numpy as jnp
import pytest

from jaxmarl.environments.cage import CageEnv, make_cage_env
from jaxmarl.environments.cage.observations import BLUE_OBS_DIM, RED_OBS_DIM
from jaxmarl.environments.cage.actions import NUM_BLUE_ACTIONS, NUM_RED_ACTIONS


@pytest.fixture
def env(scenario2_env):
    """Use session-scoped env to avoid JIT recompilation."""
    return scenario2_env


class TestEnvironmentCreation:
    def test_create_env(self):
        env = CageEnv()
        assert env.num_agents == 2
        assert "blue" in env.agents
        assert "red" in env.agents

    def test_make_cage_env(self):
        env = make_cage_env(max_steps=50)
        assert env.max_steps == 50

    def test_action_spaces(self, env):
        assert env.action_space("blue").n == NUM_BLUE_ACTIONS
        assert env.action_space("red").n == NUM_RED_ACTIONS

    def test_observation_spaces(self, env):
        assert env.observation_space("blue").shape == (BLUE_OBS_DIM,)
        assert env.observation_space("red").shape == (RED_OBS_DIM,)


class TestReset:
    def test_reset_returns_obs_and_state(self, env):
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key)

        assert "blue" in obs
        assert "red" in obs
        assert state is not None

    def test_reset_obs_shapes(self, env):
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key)

        assert obs["blue"].shape == (BLUE_OBS_DIM,)
        assert obs["red"].shape == (RED_OBS_DIM,)

    def test_reset_initial_state(self, env):
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key)

        assert state.time == 0
        assert state.done == False
        # Red starts with PRIVILEGED (SYSTEM) foothold on User0 (matches CybORG)
        from jaxmarl.environments.cage.state import HOST_IDS, COMPROMISE_PRIVILEGED
        assert state.host_compromised[HOST_IDS['User0']] == COMPROMISE_PRIVILEGED

    def test_reset_determinism(self, env):
        key = jax.random.PRNGKey(42)

        obs1, state1 = env.reset(key)
        obs2, state2 = env.reset(key)

        assert jnp.allclose(obs1["blue"], obs2["blue"])
        assert jnp.allclose(obs1["red"], obs2["red"])


class TestStep:
    def test_step_returns_correct_structure(self, env):
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key)

        actions = {"blue": jnp.array(0), "red": jnp.array(0)}  # Both sleep
        key, step_key = jax.random.split(key)

        obs, state, rewards, dones, info = env.step(step_key, state, actions)

        assert "blue" in obs
        assert "red" in obs
        assert "blue" in rewards
        assert "red" in rewards
        assert "blue" in dones
        assert "red" in dones
        assert "__all__" in dones

    def test_step_increments_time(self, env):
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key)

        actions = {"blue": jnp.array(0), "red": jnp.array(0)}
        key, step_key = jax.random.split(key)

        _, new_state, _, _, _ = env.step(step_key, state, actions)

        assert new_state.time == 1

    def test_step_terminates_at_max_steps(self, env):
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key)

        actions = {"blue": jnp.array(0), "red": jnp.array(0)}
        done_at_100 = False

        for i in range(100):
            key, step_key = jax.random.split(key)
            obs, state, rewards, dones, info = env.step(step_key, state, actions)
            if i == 99:  # Check done flag before auto-reset
                done_at_100 = dones["__all__"]

        assert done_at_100
        # After auto-reset, state.time is 0
        assert state.time == 0


class TestActionMasking:
    def test_get_avail_actions_returns_masks(self, env):
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key)

        avail = env.get_avail_actions(state)

        assert "blue" in avail
        assert "red" in avail
        assert avail["blue"].shape == (NUM_BLUE_ACTIONS,)
        assert avail["red"].shape == (NUM_RED_ACTIONS,)

    def test_action_masks_are_boolean(self, env):
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key)

        avail = env.get_avail_actions(state)

        assert avail["blue"].dtype == jnp.bool_
        assert avail["red"].dtype == jnp.bool_


class TestEpisodeRollout:
    def test_100_step_rollout(self, env):
        """Run 100 random steps without error."""
        key = jax.random.PRNGKey(42)
        obs, state = env.reset(key)

        for _ in range(100):
            key, key_blue, key_red, key_step = jax.random.split(key, 4)

            # Sample valid actions
            avail = env.get_avail_actions(state)
            blue_action = jax.random.choice(
                key_blue,
                jnp.arange(NUM_BLUE_ACTIONS),
                p=avail["blue"].astype(jnp.float32) / jnp.sum(avail["blue"]),
            )
            red_action = jax.random.choice(
                key_red,
                jnp.arange(NUM_RED_ACTIONS),
                p=avail["red"].astype(jnp.float32) / jnp.sum(avail["red"]),
            )

            actions = {"blue": blue_action, "red": red_action}
            obs, state, rewards, dones, info = env.step(key_step, state, actions)

            if dones["__all__"]:
                break

    def test_random_rollout_with_auto_reset(self, env):
        """Test that auto-reset works in step()."""
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key)

        actions = {"blue": jnp.array(0), "red": jnp.array(0)}

        # Run past episode boundary
        for i in range(150):
            key, step_key = jax.random.split(key)
            obs, state, rewards, dones, info = env.step(step_key, state, actions)

        # Should have auto-reset
        assert state.time < 100


class TestDeterminism:
    def test_same_seed_same_trajectory(self, env):
        """Same seed should produce identical trajectories."""
        def run_episode(key):
            obs, state = env.reset(key)
            total_reward = 0.0

            actions = {"blue": jnp.array(0), "red": jnp.array(0)}
            for _ in range(10):
                key, step_key = jax.random.split(key)
                obs, state, rewards, dones, info = env.step(step_key, state, actions)
                total_reward += rewards["blue"]

            return total_reward, state.host_compromised

        key1 = jax.random.PRNGKey(42)
        key2 = jax.random.PRNGKey(42)

        reward1, compromised1 = run_episode(key1)
        reward2, compromised2 = run_episode(key2)

        assert reward1 == reward2
        assert jnp.array_equal(compromised1, compromised2)


class TestJITCompilation:
    def test_reset_jit(self, env):
        """reset should JIT compile."""
        reset_jit = jax.jit(env.reset)
        key = jax.random.PRNGKey(0)

        obs, state = reset_jit(key)
        assert obs["blue"].shape == (BLUE_OBS_DIM,)

    def test_step_jit(self, env):
        """step should JIT compile."""
        step_jit = jax.jit(env.step)
        key = jax.random.PRNGKey(0)

        obs, state = env.reset(key)
        actions = {"blue": jnp.array(0), "red": jnp.array(0)}

        key, step_key = jax.random.split(key)
        obs, state, rewards, dones, info = step_jit(step_key, state, actions)

        assert state.time == 1

    def test_vmap_reset(self, env):
        """reset should work with vmap for parallel environments."""
        reset_vmap = jax.vmap(env.reset)
        keys = jax.random.split(jax.random.PRNGKey(0), 32)

        obs, states = reset_vmap(keys)

        assert obs["blue"].shape == (32, BLUE_OBS_DIM)
        assert obs["red"].shape == (32, RED_OBS_DIM)


class TestRewards:
    def test_rewards_are_scalars(self, env):
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key)

        actions = {"blue": jnp.array(0), "red": jnp.array(0)}
        key, step_key = jax.random.split(key)

        _, _, rewards, _, _ = env.step(step_key, state, actions)

        assert rewards["blue"].shape == ()
        assert rewards["red"].shape == ()

    def test_cumulative_rewards_tracked(self, env):
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key)

        actions = {"blue": jnp.array(0), "red": jnp.array(0)}

        for _ in range(10):
            key, step_key = jax.random.split(key)
            obs, state, rewards, dones, info = env.step(step_key, state, actions)

        assert "cumulative_red_reward" in info
        assert "cumulative_blue_reward" in info


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
