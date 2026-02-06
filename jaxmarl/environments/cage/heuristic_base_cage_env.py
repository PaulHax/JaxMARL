"""Shared base for CAGE heuristic Red wrappers.

Centralizes the common single-player wrapper behavior for Blue agents
while scripted Red policies are managed by subclasses.
"""

import jax
import chex
from typing import Dict, Tuple
from functools import partial

from jaxmarl.environments.multi_agent_env import MultiAgentEnv
from jaxmarl.environments.cage.cage_env import CageEnv
from jaxmarl.environments.cage.state import CageState


class HeuristicBaseCAGE(MultiAgentEnv):
    """Base wrapper for scripted-Red single-player CAGE environments."""

    def __init__(
        self,
        scenario: str = "Scenario2",
        max_steps: int = 100,
        num_blue_agents: int = 1,
        **kwargs
    ):
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

    def _blue_obs(self, obs: Dict[str, chex.Array]) -> Dict[str, chex.Array]:
        return {agent: obs[agent] for agent in self.agents}

    def _blue_outputs(
        self,
        obs: Dict[str, chex.Array],
        rewards: Dict[str, chex.Array],
        dones: Dict[str, chex.Array],
    ) -> Tuple[Dict[str, chex.Array], Dict[str, chex.Array], Dict[str, chex.Array]]:
        blue_obs = self._blue_obs(obs)
        blue_rewards = {agent: rewards[agent] for agent in self.agents}
        blue_dones = {agent: dones[agent] for agent in self.agents}
        blue_dones['__all__'] = dones['__all__']
        return blue_obs, blue_rewards, blue_dones

    def _step_env_with_red_action(
        self,
        key: chex.PRNGKey,
        env_state: CageState,
        actions: Dict[str, chex.Array],
        red_action: chex.Array,
    ):
        full_actions = {**actions, self._env.red_agents[0]: red_action}
        return self._env.step_env(key, env_state, full_actions)

    @partial(jax.jit, static_argnums=(0,))
    def get_obs(self, state) -> Dict[str, chex.Array]:
        """Get observations for blue agents only."""
        obs = self._env.get_obs(state.state)
        return self._blue_obs(obs)

    @partial(jax.jit, static_argnums=(0,))
    def get_avail_actions(self, state) -> Dict[str, chex.Array]:
        """Get available action masks for blue agents only."""
        avail = self._env.get_avail_actions(state.state)
        return {agent: avail[agent] for agent in self.agents}

    def observation_space(self, agent: str):
        """Get observation space for an agent."""
        return self.observation_spaces[agent]

    def action_space(self, agent: str):
        """Get action space for an agent."""
        return self.action_spaces[agent]
