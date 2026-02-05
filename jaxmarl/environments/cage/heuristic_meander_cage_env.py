"""CAGE environment wrapper with scripted RedMeanderAgent as Red opponent.

Follows the HeuristicEnemySMAX pattern: presents CAGE as a single-player
(but still multi-agent for Blue team) environment where Blue trains
against a fixed scripted Red strategy.

Unlike HeuristicRedCAGE which uses B_lineAgent (deterministic FSM),
this uses RedMeanderAgent which explores the network opportunistically.
"""

import jax
import chex
from typing import Dict, Tuple
from flax import struct
from functools import partial

from jaxmarl.environments.cage.heuristic_base_cage_env import HeuristicBaseCAGE
from jaxmarl.environments.cage.state import CageState
from jaxmarl.environments.cage.observations import get_red_obs
from jaxmarl.environments.cage.actions import get_red_action_mask
from jaxmarl.environments.cage.scripted_agents import (
    MeanderState,
    meander_reset,
    meander_get_action,
)


@struct.dataclass
class HeuristicMeanderState:
    """Combined state for underlying CAGE env and RedMeanderAgent."""
    state: CageState
    red_policy_state: MeanderState


class HeuristicMeanderCAGE(HeuristicBaseCAGE):
    """Single-player CAGE where Blue trains against scripted RedMeanderAgent.

    Wraps CageEnv and internally manages the Red agent using RedMeanderAgent,
    an exploratory/opportunistic agent that scans all discovered hosts,
    exploits available targets, and escalates privileges where possible.

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
        """Initialize HeuristicMeanderCAGE environment.

        Args:
            scenario: Scenario name (Scenario2, hosts_2, hosts_3, hosts_4, hosts_5)
            max_steps: Maximum episode length
            num_blue_agents: Number of blue team agents
            **kwargs: Additional arguments passed to CageEnv
        """
        super().__init__(
            scenario=scenario,
            max_steps=max_steps,
            num_blue_agents=num_blue_agents,
            **kwargs
        )

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], HeuristicMeanderState]:
        """Reset environment and RedMeanderAgent state."""
        key, env_key, meander_key = jax.random.split(key, 3)
        obs, env_state = self._env.reset(env_key)
        red_policy_state = meander_reset(self.const, meander_key)

        blue_obs = self._blue_obs(obs)

        return blue_obs, HeuristicMeanderState(
            state=env_state,
            red_policy_state=red_policy_state
        )

    @partial(jax.jit, static_argnums=(0,))
    def step_env(
        self,
        key: chex.PRNGKey,
        state: HeuristicMeanderState,
        actions: Dict[str, chex.Array],
    ) -> Tuple[Dict[str, chex.Array], HeuristicMeanderState, Dict[str, float], Dict[str, bool], Dict]:
        """Execute one environment step with RedMeanderAgent as Red.

        Args:
            key: Random key
            state: Combined environment and RedMeanderAgent state
            actions: Blue agent actions only

        Returns:
            Tuple of (blue_obs, new_state, blue_rewards, blue_dones, info)
        """
        key, red_key, step_key, reset_key = jax.random.split(key, 4)

        red_obs = get_red_obs(state.state, self.const)
        red_mask = get_red_action_mask(state.state, self.const)

        red_action, new_red_policy_state = meander_get_action(
            state.red_policy_state,
            red_obs,
            red_mask,
            self.const,
            red_key,
            state.state.host_services,
        )

        obs, new_env_state, rewards, dones, info = self._step_env_with_red_action(
            step_key, state.state, actions, red_action
        )

        new_red_policy_state = jax.lax.cond(
            dones['__all__'],
            lambda k: meander_reset(self.const, k),
            lambda _: new_red_policy_state,
            reset_key
        )

        blue_obs, blue_rewards, blue_dones = self._blue_outputs(obs, rewards, dones)

        new_state = HeuristicMeanderState(
            state=new_env_state,
            red_policy_state=new_red_policy_state
        )

        return blue_obs, new_state, blue_rewards, blue_dones, info

    @property
    def name(self) -> str:
        return f"HeuristicMeanderCAGE-{self.scenario}"

    def get_scenario_info(self) -> dict:
        """Get information about the current scenario configuration."""
        info = self._env.get_scenario_info()
        info["red_agent"] = "RedMeanderAgent"
        info["blue_only"] = True
        return info
