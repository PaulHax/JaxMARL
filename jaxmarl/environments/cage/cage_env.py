"""CAGE-JAX: JAX implementation of CAGE Challenge 2 environment with configurable support."""

import jax
import jax.numpy as jnp
import chex
from functools import partial
from typing import Dict, Tuple, Optional, List

from jaxmarl.environments.multi_agent_env import MultiAgentEnv
from jaxmarl.environments.spaces import Discrete, Box

from jaxmarl.environments.cage.config import (
    ScenarioConfig, create_scenario2_config, get_scenario, create_multi_agent_config
)
from jaxmarl.environments.cage.state import (
    CageState, CageConst,
    create_scenario2_const, create_const_from_scenario,
    create_initial_state_with_red_foothold, build_const_from_config,
)
from jaxmarl.environments.cage.actions import (
    apply_blue_action, apply_red_action,
    get_blue_action_mask, get_red_action_mask,
    compute_blue_action_space_size, compute_red_action_space_size,
    NUM_BLUE_ACTIONS, NUM_RED_ACTIONS,
)
from jaxmarl.environments.cage.observations import (
    get_blue_obs, get_red_obs,
    compute_blue_obs_dim, compute_red_obs_dim,
    BLUE_OBS_DIM, RED_OBS_DIM,
)
from jaxmarl.environments.cage.rewards import compute_rewards


class CageEnv(MultiAgentEnv):
    """JAX implementation of CAGE Challenge 2 environment.

    A multi-agent competitive environment where:
    - Blue (defender) agents try to protect a network from Red attacks
    - Red (attacker) agents try to compromise hosts and reach operational servers

    Features:
    - Fully JIT-compilable for fast parallel simulation
    - Compatible with JaxMARL training infrastructure
    - Supports configurable scenarios with variable host counts
    - Supports multiple agents per team
    """

    def __init__(
        self,
        scenario: str = "Scenario2",
        max_steps: int = 100,
        config: Optional[ScenarioConfig] = None,
        num_red_agents: int = 1,
        num_blue_agents: int = 1,
    ):
        """Initialize CAGE environment.

        Args:
            scenario: Scenario name (Scenario2, hosts_2, hosts_3, hosts_4, hosts_5)
            max_steps: Maximum episode length (default 100)
            config: Optional ScenarioConfig to use instead of named scenario
            num_red_agents: Number of red team agents (default 1)
            num_blue_agents: Number of blue team agents (default 1)
        """
        # Build configuration
        if config is not None:
            self.config = config
        else:
            self.config = get_scenario(scenario)

        # Handle multi-agent configuration
        if num_red_agents > 1 or num_blue_agents > 1:
            self.config = create_multi_agent_config(
                num_red_agents=num_red_agents,
                num_blue_agents=num_blue_agents,
                base_config=self.config,
            )

        self.const = build_const_from_config(self.config)
        self.max_steps = max_steps

        # Build agent lists
        self.num_red_agents = self.config.num_red_agents
        self.num_blue_agents = self.config.num_blue_agents
        num_agents = self.num_red_agents + self.num_blue_agents

        super().__init__(num_agents=num_agents)

        # Build agent names
        if self.num_blue_agents == 1:
            self.blue_agents = ["blue"]
        else:
            self.blue_agents = [f"blue_{i}" for i in range(self.num_blue_agents)]

        if self.num_red_agents == 1:
            self.red_agents = ["red"]
        else:
            self.red_agents = [f"red_{i}" for i in range(self.num_red_agents)]

        self.agents = self.blue_agents + self.red_agents
        self.agent_ids = {name: i for i, name in enumerate(self.agents)}

        # Compute action/observation sizes
        self.blue_action_size = compute_blue_action_space_size(self.const)
        self.red_action_size = compute_red_action_space_size(self.const)
        self.blue_obs_dim = compute_blue_obs_dim(self.const)
        self.red_obs_dim = compute_red_obs_dim(self.const)

        # Build action spaces
        self.action_spaces = {}
        for agent in self.blue_agents:
            self.action_spaces[agent] = Discrete(self.blue_action_size)
        for agent in self.red_agents:
            self.action_spaces[agent] = Discrete(self.red_action_size)

        # Build observation spaces
        self.observation_spaces = {}
        for agent in self.blue_agents:
            self.observation_spaces[agent] = Box(
                low=0.0, high=1.0, shape=(self.blue_obs_dim,), dtype=jnp.float32
            )
        for agent in self.red_agents:
            self.observation_spaces[agent] = Box(
                low=0.0, high=1.0, shape=(self.red_obs_dim,), dtype=jnp.float32
            )

    @partial(jax.jit, static_argnums=[0])
    def reset(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], CageState]:
        """Reset environment to initial state."""
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
        """Execute one environment step."""
        # Aggregate blue actions (for simplicity, use first blue agent's action)
        # In multi-agent, each blue agent can act on different hosts
        blue_action = actions[self.blue_agents[0]]

        # Aggregate red actions (for simplicity, use first red agent's action)
        red_action = actions[self.red_agents[0]]

        key, key_red = jax.random.split(key)

        # Apply Blue action first (defender gets priority)
        state = apply_blue_action(state, blue_action, self.const)

        # Apply Red action
        state = apply_red_action(state, red_action, self.const, key_red)

        # Compute rewards
        rewards = compute_rewards(state, self.const, blue_action, red_action)

        # Distribute rewards to all agents on each team
        all_rewards = {}
        for agent in self.blue_agents:
            all_rewards[agent] = rewards["blue"]
        for agent in self.red_agents:
            all_rewards[agent] = rewards["red"]

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

        dones = {agent: done for agent in self.agents}
        dones["__all__"] = done

        obs = self.get_obs(state)

        info = {
            "cumulative_red_reward": state.cumulative_red_reward,
            "cumulative_blue_reward": state.cumulative_blue_reward,
        }

        return obs, state, all_rewards, dones, info

    @partial(jax.jit, static_argnums=[0])
    def get_obs(self, state: CageState) -> Dict[str, chex.Array]:
        """Get observations for all agents."""
        blue_obs = get_blue_obs(state, self.const)
        red_obs = get_red_obs(state, self.const)

        obs = {}
        for agent in self.blue_agents:
            obs[agent] = blue_obs
        for agent in self.red_agents:
            obs[agent] = red_obs

        return obs

    @partial(jax.jit, static_argnums=[0])
    def get_avail_actions(self, state: CageState) -> Dict[str, chex.Array]:
        """Get available action masks for all agents."""
        blue_mask = get_blue_action_mask(state, self.const)
        red_mask = get_red_action_mask(state, self.const)

        masks = {}
        for agent in self.blue_agents:
            masks[agent] = blue_mask
        for agent in self.red_agents:
            masks[agent] = red_mask

        return masks

    @property
    def name(self) -> str:
        return f"CAGE-{self.config.name}"

    @property
    def agent_classes(self) -> dict:
        return {
            "blue_agents": self.blue_agents,
            "red_agents": self.red_agents,
        }

    def get_scenario_info(self) -> dict:
        """Get information about the current scenario configuration."""
        return {
            "scenario_name": self.config.name,
            "num_hosts": self.const.num_hosts,
            "num_subnets": self.const.num_subnets,
            "num_services": self.const.num_services,
            "num_exploits": self.const.num_exploits,
            "num_decoys": self.const.num_decoys,
            "num_decoy_hosts": self.const.num_decoy_hosts,
            "num_red_agents": self.num_red_agents,
            "num_blue_agents": self.num_blue_agents,
            "blue_action_space_size": self.blue_action_size,
            "red_action_space_size": self.red_action_size,
            "blue_obs_dim": self.blue_obs_dim,
            "red_obs_dim": self.red_obs_dim,
            "max_steps": self.max_steps,
        }


def make_cage_env(
    scenario: str = "Scenario2",
    max_steps: int = 100,
    num_red_agents: int = 1,
    num_blue_agents: int = 1,
    **kwargs
) -> CageEnv:
    """Factory function to create CAGE environment.

    Args:
        scenario: Scenario name (Scenario2, hosts_2, hosts_3, hosts_4, hosts_5)
        max_steps: Maximum episode length
        num_red_agents: Number of red team agents
        num_blue_agents: Number of blue team agents
        **kwargs: Additional arguments passed to CageEnv

    Returns:
        CageEnv instance
    """
    return CageEnv(
        scenario=scenario,
        max_steps=max_steps,
        num_red_agents=num_red_agents,
        num_blue_agents=num_blue_agents,
        **kwargs
    )
