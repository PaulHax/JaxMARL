"""CAGE-JAX: JAX implementation of CAGE Challenge 2 environment."""

import jax
import jax.numpy as jnp
import chex
from functools import partial
from typing import Dict, Tuple, Optional

from jaxmarl.environments.multi_agent_env import MultiAgentEnv
from jaxmarl.environments.spaces import Discrete, Box

from jaxmarl.environments.cage.state import (
    CageState, CageConst,
    create_scenario2_const, create_initial_state_with_red_foothold,
    NUM_HOSTS,
)
from jaxmarl.environments.cage.actions import (
    apply_blue_action, apply_red_action,
    get_blue_action_mask, get_red_action_mask,
    NUM_BLUE_ACTIONS, NUM_RED_ACTIONS,
)
from jaxmarl.environments.cage.observations import (
    get_blue_obs, get_red_obs,
    BLUE_OBS_DIM, RED_OBS_DIM,
)
from jaxmarl.environments.cage.rewards import compute_rewards


class CageEnv(MultiAgentEnv):
    """JAX implementation of CAGE Challenge 2 environment.

    A two-player competitive environment where:
    - Blue (defender) tries to protect a network from Red attacks
    - Red (attacker) tries to compromise hosts and reach the operational server

    Features:
    - Fully JIT-compilable for fast parallel simulation
    - Compatible with JaxMARL training infrastructure
    - Implements Scenario 2 from CAGE Challenge 2
    """

    def __init__(self, max_steps: int = 100):
        """Initialize CAGE environment.

        Args:
            max_steps: Maximum episode length (default 100)
        """
        super().__init__(num_agents=2)

        self.agents = ["blue", "red"]
        self.agent_ids = {"blue": 0, "red": 1}

        # Load scenario configuration
        self.const = create_scenario2_const()
        self.max_steps = max_steps

        # Action spaces
        self.action_spaces = {
            "blue": Discrete(NUM_BLUE_ACTIONS),
            "red": Discrete(NUM_RED_ACTIONS),
        }

        # Observation spaces
        self.observation_spaces = {
            "blue": Box(low=0.0, high=1.0, shape=(BLUE_OBS_DIM,), dtype=jnp.float32),
            "red": Box(low=0.0, high=1.0, shape=(RED_OBS_DIM,), dtype=jnp.float32),
        }

    @partial(jax.jit, static_argnums=[0])
    def reset(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], CageState]:
        """Reset environment to initial state.

        Args:
            key: JAX PRNG key

        Returns:
            obs: Dictionary of observations for each agent
            state: Initial environment state
        """
        state = create_initial_state_with_red_foothold(self.const)
        obs = self.get_obs(state)
        return obs, state

    @partial(jax.jit, static_argnums=[0])
    def step_env(
        self,
        key: chex.PRNGKey,
        state: CageState,
        actions: Dict[str, chex.Array],
    ) -> Tuple[Dict[str, chex.Array], CageState, Dict[str, float], Dict[str, bool], Dict]:
        """Execute one environment step.

        Args:
            key: JAX PRNG key
            state: Current environment state
            actions: Dictionary of actions for each agent

        Returns:
            obs: Next observations
            state: Next state
            rewards: Rewards for each agent
            dones: Done flags for each agent
            info: Additional info
        """
        blue_action = actions["blue"]
        red_action = actions["red"]

        # Split key for stochastic actions
        key, key_red = jax.random.split(key)

        # Apply Blue action first (defender gets priority)
        state = apply_blue_action(state, blue_action, self.const)

        # Apply Red action
        state = apply_red_action(state, red_action, self.const, key_red)

        # Compute rewards
        rewards = compute_rewards(state, self.const, blue_action, red_action)

        # Update cumulative rewards
        state = state.replace(
            cumulative_red_reward=state.cumulative_red_reward + rewards["red"],
            cumulative_blue_reward=state.cumulative_blue_reward + rewards["blue"],
        )

        # Increment time
        state = state.replace(time=state.time + 1)

        # Check termination
        done = state.time >= self.max_steps
        state = state.replace(done=jnp.array(done))

        dones = {
            "blue": done,
            "red": done,
            "__all__": done,
        }

        # Get observations
        obs = self.get_obs(state)

        info = {
            "cumulative_red_reward": state.cumulative_red_reward,
            "cumulative_blue_reward": state.cumulative_blue_reward,
        }

        return obs, state, rewards, dones, info

    @partial(jax.jit, static_argnums=[0])
    def get_obs(self, state: CageState) -> Dict[str, chex.Array]:
        """Get observations for all agents.

        Args:
            state: Current environment state

        Returns:
            Dictionary of observations keyed by agent name
        """
        return {
            "blue": get_blue_obs(state),
            "red": get_red_obs(state),
        }

    @partial(jax.jit, static_argnums=[0])
    def get_avail_actions(self, state: CageState) -> Dict[str, chex.Array]:
        """Get available action masks for all agents.

        Args:
            state: Current environment state

        Returns:
            Dictionary of boolean action masks keyed by agent name
        """
        return {
            "blue": get_blue_action_mask(state, self.const),
            "red": get_red_action_mask(state, self.const),
        }

    @property
    def name(self) -> str:
        return "CAGE"

    @property
    def agent_classes(self) -> dict:
        return {
            "agents": ["blue", "red"],
        }


def make_cage_env(**kwargs) -> CageEnv:
    """Factory function to create CAGE environment.

    Args:
        **kwargs: Keyword arguments passed to CageEnv

    Returns:
        CageEnv instance
    """
    return CageEnv(**kwargs)
