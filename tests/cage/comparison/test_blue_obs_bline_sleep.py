"""Blue observation parity tests for Monitor gating vs Blue Sleep.

These capture the Monitor-gating behavior: Blue should only observe activity
when it runs Monitor. We compare CybORG's BlueTableWrapper vector output
against JAX CAGE observations under identical Red actions.
"""

import inspect
import random

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from jaxmarl.environments.cage.cage_env import CageEnv
from tests.cage.differential.action_translator import (
    cyborg_red_action_to_jax,
    jax_action_to_cyborg,
)
from tests.cage.differential.harness import is_cyborg_available


requires_cyborg = pytest.mark.skipif(
    not is_cyborg_available(),
    reason="CybORG not installed"
)


def _assert_blue_obs_parity(blue_action_idx: int, steps: int = 20, seed: int = 42):
    random.seed(seed)

    from CybORG import CybORG
    from CybORG.Agents.SimpleAgents.B_line import B_lineAgent
    from CybORG.Agents.Wrappers import BlueTableWrapper

    path = str(inspect.getfile(CybORG))
    path = path[:-10] + "/Shared/Scenarios/Scenario2.yaml"
    cyborg = CybORG(path, "sim", agents={"Red": B_lineAgent})

    cyborg.reset()
    blue_wrapper = BlueTableWrapper(env=cyborg, output_mode="vector")
    blue_wrapper.reset("Blue")
    red_agent = B_lineAgent()

    jax_env = CageEnv(scenario="Scenario2")
    key = jax.random.PRNGKey(seed)
    obs_jax, state_jax = jax_env.reset(key)

    blue_action = jax_action_to_cyborg(blue_action_idx, cyborg, "Blue")

    for step in range(steps):
        obs_red = cyborg.get_observation("Red")
        action_space_red = cyborg.get_action_space("Red")
        cyborg.step("Blue", blue_action)
        red_action = red_agent.get_action(obs_red, action_space_red)
        cyborg.step("Red", red_action)
        raw_obs_cyborg = cyborg.get_observation("Blue")
        obs_cyborg = blue_wrapper.observation_change(raw_obs_cyborg)

        red_action_jax = cyborg_red_action_to_jax(red_action, cyborg, "Red")

        key, step_key = jax.random.split(key)
        obs_jax, state_jax, rewards, dones, info = jax_env.step_env(
            step_key,
            state_jax,
            {"blue": jnp.array(blue_action_idx), "red": jnp.array(red_action_jax)},
        )

        assert np.allclose(obs_jax["blue"], obs_cyborg, atol=1e-5), (
            f"Blue obs mismatch at step {step}"
        )


@requires_cyborg
def test_blue_obs_bline_sleep_parity():
    from CybORG import CybORG
    _assert_blue_obs_parity(blue_action_idx=0)


@requires_cyborg
def test_blue_obs_bline_monitor_parity():
    _assert_blue_obs_parity(blue_action_idx=1)
