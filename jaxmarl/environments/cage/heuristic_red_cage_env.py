"""CAGE environment wrapper with scripted B_lineAgent as Red opponent.

Follows the HeuristicEnemySMAX pattern: presents CAGE as a single-player
(but still multi-agent for Blue team) environment where Blue trains
against a fixed scripted Red strategy.
"""

import jax
import jax.numpy as jnp
import chex
from typing import Dict, Tuple
from flax import struct
from functools import partial

from jaxmarl.environments.multi_agent_env import MultiAgentEnv
from jaxmarl.environments.cage.cage_env import CageEnv
from jaxmarl.environments.cage.state import CageState, CageConst
from jaxmarl.environments.cage.observations import get_red_obs
from jaxmarl.environments.cage.actions import get_red_action_mask
from jaxmarl.environments.cage.scripted_agents import (
    BLineState,
    bline_reset,
    bline_get_action,
)


@struct.dataclass
class HeuristicRedState:
    """Combined state for underlying CAGE env and B_lineAgent FSM."""
    state: CageState
    red_policy_state: BLineState


class HeuristicRedCAGE(MultiAgentEnv):
    """Single-player CAGE where Blue trains against scripted B_lineAgent.

    Wraps CageEnv and internally manages the Red agent using B_lineAgent,
    a deterministic finite state machine that follows the CAGE Challenge 2
    attack path (User1 -> Enterprise0 -> Enterprise2 -> Op_Server0).

    Only Blue agents are exposed to the training algorithm. Red observations,
    actions, and rewards are handled internally.
    """

    def __init__(
        self,
        scenario: str = "Scenario2",
        max_steps: int = 100,
        num_blue_agents: int = 1,
        **kwargs
    ):
        """Initialize HeuristicRedCAGE environment.

        Args:
            scenario: Scenario name (Scenario2, hosts_2, hosts_3, hosts_4, hosts_5)
            max_steps: Maximum episode length
            num_blue_agents: Number of blue team agents
            **kwargs: Additional arguments passed to CageEnv
        """
        self._env = CageEnv(
            scenario=scenario,
            max_steps=max_steps,
            num_blue_agents=num_blue_agents,
            num_red_agents=1,
            **kwargs
        )

        self.num_agents = self._env.num_blue_agents
        self.agents = self._env.blue_agents
        self.num_blue_agents = self._env.num_blue_agents

        self.observation_spaces = {
            agent: self._env.observation_spaces[agent]
            for agent in self.agents
        }
        self.action_spaces = {
            agent: self._env.action_spaces[agent]
            for agent in self.agents
        }

        self.const = self._env.const
        self.max_steps = max_steps
        self.scenario = scenario

    def __getattr__(self, name: str):
        return getattr(self._env, name)

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], HeuristicRedState]:
        """Reset environment and B_lineAgent state."""
        obs, env_state = self._env.reset(key)
        red_policy_state = bline_reset()

        blue_obs = {agent: obs[agent] for agent in self.agents}

        return blue_obs, HeuristicRedState(
            state=env_state,
            red_policy_state=red_policy_state
        )

    @partial(jax.jit, static_argnums=(0,))
    def step_env(
        self,
        key: chex.PRNGKey,
        state: HeuristicRedState,
        actions: Dict[str, chex.Array],
    ) -> Tuple[Dict[str, chex.Array], HeuristicRedState, Dict[str, float], Dict[str, bool], Dict]:
        """Execute one environment step with B_lineAgent as Red.

        Args:
            key: Random key
            state: Combined environment and B_lineAgent state
            actions: Blue agent actions only

        Returns:
            Tuple of (blue_obs, new_state, blue_rewards, blue_dones, info)
        """
        key, red_key, step_key = jax.random.split(key, 3)

        red_obs = get_red_obs(state.state, self.const)
        red_mask = get_red_action_mask(state.state, self.const)

        red_action, new_red_policy_state = bline_get_action(
            state.red_policy_state,
            red_obs,
            red_mask,
            self.const,
            red_key
        )

        full_actions = {**actions, self._env.red_agents[0]: red_action}

        obs, new_env_state, rewards, dones, info = self._env.step_env(
            step_key, state.state, full_actions
        )

        new_red_policy_state = jax.lax.cond(
            dones['__all__'],
            lambda _: bline_reset(),
            lambda s: s,
            new_red_policy_state
        )

        blue_obs = {agent: obs[agent] for agent in self.agents}
        blue_rewards = {agent: rewards[agent] for agent in self.agents}
        blue_dones = {agent: dones[agent] for agent in self.agents}
        blue_dones['__all__'] = dones['__all__']

        new_state = HeuristicRedState(
            state=new_env_state,
            red_policy_state=new_red_policy_state
        )

        return blue_obs, new_state, blue_rewards, blue_dones, info

    @partial(jax.jit, static_argnums=(0,))
    def get_obs(self, state: HeuristicRedState) -> Dict[str, chex.Array]:
        """Get observations for blue agents only."""
        obs = self._env.get_obs(state.state)
        return {agent: obs[agent] for agent in self.agents}

    @partial(jax.jit, static_argnums=(0,))
    def get_avail_actions(self, state: HeuristicRedState) -> Dict[str, chex.Array]:
        """Get available action masks for blue agents only."""
        avail = self._env.get_avail_actions(state.state)
        return {agent: avail[agent] for agent in self.agents}

    def observation_space(self, agent: str):
        """Get observation space for an agent."""
        return self.observation_spaces[agent]

    def action_space(self, agent: str):
        """Get action space for an agent."""
        return self.action_spaces[agent]

    @property
    def name(self) -> str:
        return f"HeuristicRedCAGE-{self.scenario}"

    def get_scenario_info(self) -> dict:
        """Get information about the current scenario configuration."""
        info = self._env.get_scenario_info()
        info["red_agent"] = "B_lineAgent"
        info["blue_only"] = True
        return info
