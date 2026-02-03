"""CAGE environment wrapper with scripted Red opponent.

Follows the HeuristicEnemySMAX pattern: presents CAGE as a single-player
(but still multi-agent for Blue team) environment where Blue trains
against a fixed scripted Red strategy.

Supports multiple Red agent types:
- "bline": B_lineAgent - deterministic FSM following fixed attack path
- "meander": RedMeanderAgent - exploratory/opportunistic agent
"""

import jax
import jax.numpy as jnp
import chex
from typing import Dict, Tuple, Union, Literal
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
    MeanderState,
    meander_reset,
    meander_get_action,
)


RedAgentType = Literal["bline", "meander"]


@struct.dataclass
class HeuristicRedState:
    """Combined state for underlying CAGE env and scripted Red agent."""
    state: CageState
    bline_state: BLineState
    meander_state: MeanderState


class HeuristicRedCAGE(MultiAgentEnv):
    """Single-player CAGE where Blue trains against a scripted Red agent.

    Wraps CageEnv and internally manages the Red agent using one of:
    - B_lineAgent ("bline"): Deterministic FSM following CAGE Challenge 2
      attack path (User1 -> Enterprise1 -> Enterprise2 -> Op_Server0)
    - RedMeanderAgent ("meander"): Exploratory agent that scans hosts in
      random order and exploits opportunistically

    Only Blue agents are exposed to the training algorithm. Red observations,
    actions, and rewards are handled internally.
    """

    def __init__(
        self,
        scenario: str = "Scenario2",
        max_steps: int = 100,
        num_blue_agents: int = 1,
        red_agent: RedAgentType = "bline",
        **kwargs
    ):
        """Initialize HeuristicRedCAGE environment.

        Args:
            scenario: Scenario name (Scenario2, hosts_2, hosts_3, hosts_4, hosts_5)
            max_steps: Maximum episode length
            num_blue_agents: Number of blue team agents
            red_agent: Type of Red agent ("bline" or "meander")
            **kwargs: Additional arguments passed to CageEnv
        """
        if red_agent not in ("bline", "meander"):
            raise ValueError(f"red_agent must be 'bline' or 'meander', got {red_agent!r}")

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
        self.red_agent = red_agent

    def __getattr__(self, name: str):
        return getattr(self._env, name)

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], HeuristicRedState]:
        """Reset environment and Red agent state."""
        key, env_key, red_key = jax.random.split(key, 3)
        obs, env_state = self._env.reset(env_key)

        bline_state = bline_reset(red_key)
        meander_state = meander_reset(self.const, red_key)

        blue_obs = {agent: obs[agent] for agent in self.agents}

        return blue_obs, HeuristicRedState(
            state=env_state,
            bline_state=bline_state,
            meander_state=meander_state,
        )

    @partial(jax.jit, static_argnums=(0,))
    def step_env(
        self,
        key: chex.PRNGKey,
        state: HeuristicRedState,
        actions: Dict[str, chex.Array],
    ) -> Tuple[Dict[str, chex.Array], HeuristicRedState, Dict[str, float], Dict[str, bool], Dict]:
        """Execute one environment step with scripted Red agent.

        Args:
            key: Random key
            state: Combined environment and Red agent state
            actions: Blue agent actions only

        Returns:
            Tuple of (blue_obs, new_state, blue_rewards, blue_dones, info)
        """
        key, red_key, step_key, reset_key = jax.random.split(key, 4)

        red_obs = get_red_obs(state.state, self.const)
        red_mask = get_red_action_mask(state.state, self.const)

        if self.red_agent == "bline":
            red_action, new_bline_state = bline_get_action(
                state.bline_state, red_obs, red_mask, self.const, red_key
            )
            new_meander_state = state.meander_state
        else:
            red_action, new_meander_state = meander_get_action(
                state.meander_state, red_obs, red_mask, self.const, red_key
            )
            new_bline_state = state.bline_state

        full_actions = {**actions, self._env.red_agents[0]: red_action}

        obs, new_env_state, rewards, dones, info = self._env.step_env(
            step_key, state.state, full_actions
        )

        if self.red_agent == "bline":
            new_bline_state = jax.lax.cond(
                dones['__all__'],
                lambda k: bline_reset(k),
                lambda _: new_bline_state,
                reset_key
            )
        else:
            new_meander_state = jax.lax.cond(
                dones['__all__'],
                lambda k: meander_reset(self.const, k),
                lambda _: new_meander_state,
                reset_key
            )

        blue_obs = {agent: obs[agent] for agent in self.agents}
        blue_rewards = {agent: rewards[agent] for agent in self.agents}
        blue_dones = {agent: dones[agent] for agent in self.agents}
        blue_dones['__all__'] = dones['__all__']

        new_state = HeuristicRedState(
            state=new_env_state,
            bline_state=new_bline_state,
            meander_state=new_meander_state,
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
        agent_name = "BLine" if self.red_agent == "bline" else "Meander"
        return f"HeuristicRedCAGE-{agent_name}-{self.scenario}"

    def get_scenario_info(self) -> dict:
        """Get information about the current scenario configuration."""
        info = self._env.get_scenario_info()
        info["red_agent"] = "B_lineAgent" if self.red_agent == "bline" else "RedMeanderAgent"
        info["blue_only"] = True
        return info


# Backward compatibility alias
HeuristicMeanderCAGE = lambda **kwargs: HeuristicRedCAGE(red_agent="meander", **kwargs)
