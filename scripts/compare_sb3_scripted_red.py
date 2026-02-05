#!/usr/bin/env python3
"""Compare SB3 policy behavior in CybORG vs JAX with scripted Red actions.

This isolates observation drift under identical Red actions.
"""
import argparse
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import jax
import jax.numpy as jnp

from jaxmarl.environments.cage import HeuristicRedCAGE
from tests.cage.differential.action_translator import (
    cyborg_red_action_to_jax,
    jax_action_to_cyborg,
)
from tests.cage.differential.harness import is_cyborg_available


HOST_NAMES = [
    "Defender", "Enterprise0", "Enterprise1", "Enterprise2",
    "Op_Host0", "Op_Host1", "Op_Host2", "Op_Server0",
    "User0", "User1", "User2", "User3", "User4"
]
FEATURE_NAMES = ["activity_0", "activity_1", "compromised_0", "compromised_1"]


def format_blue_obs_diffs(jax_obs, cyborg_obs, tolerance: float = 1e-5) -> str:
    jax_obs_np = np.array(jax_obs)
    cyborg_obs_np = np.array(cyborg_obs)

    if jax_obs_np.shape != cyborg_obs_np.shape or jax_obs_np.size % 4 != 0:
        return "  (obs shape mismatch, cannot decode per-host features)"

    diff_indices = np.where(np.abs(jax_obs_np - cyborg_obs_np) > tolerance)[0]
    if diff_indices.size == 0:
        return "  (no feature-level diffs)"

    by_host = {}
    for idx in diff_indices:
        host_idx = idx // 4
        feat_idx = idx % 4
        host_name = HOST_NAMES[host_idx] if host_idx < len(HOST_NAMES) else f"host_{host_idx}"
        feat_name = FEATURE_NAMES[feat_idx]
        by_host.setdefault(host_name, []).append(
            f"{feat_name}: JAX={jax_obs_np[idx]:.2f} CybORG={cyborg_obs_np[idx]:.2f}"
        )

    lines = []
    for host_name in sorted(by_host.keys()):
        feats = "; ".join(by_host[host_name])
        lines.append(f"  Host {host_name}: {feats}")
    return "\n".join(lines)


def load_sb3(model_path: str):
    try:
        try:
            import gymnasium as gym  # noqa: F401
            import sys as _sys
            if "gym" in _sys.modules:
                del _sys.modules["gym"]
            _sys.modules["gym"] = gym
            _sys.modules["gym.spaces"] = gym.spaces
            _sys.modules["gym.wrappers"] = gym.wrappers
        except ImportError:
            pass
        from stable_baselines3 import PPO
        return PPO.load(model_path, device="cpu")
    except Exception as exc:
        raise SystemExit(f"Failed to load SB3 policy: {exc}")


def precompute_red_sequence(cyborg_env, red_agent, steps: int):
    """Run CybORG with Blue Sleep to capture a red action sequence."""
    red_actions = []
    obs = cyborg_env.get_observation("Red")
    action_space = cyborg_env.get_action_space("Red")
    blue_sleep = jax_action_to_cyborg(0, cyborg_env, "Blue")
    for _ in range(steps):
        cyborg_action = red_agent.get_action(obs, action_space)
        red_actions.append(cyborg_action)
        # Step Blue Sleep (action 0) then Red action
        cyborg_env.step("Blue", blue_sleep)
        cyborg_env.step("Red", cyborg_action)
        obs = cyborg_env.get_observation("Red")
    return red_actions


def main():
    if not is_cyborg_available():
        raise SystemExit("CybORG not available; cannot run comparison.")

    parser = argparse.ArgumentParser(description="Compare SB3 in CybORG vs JAX under scripted Red actions")
    parser.add_argument("--model", required=True, help="Path to SB3 .zip model")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from CybORG import CybORG
    from CybORG.Agents.SimpleAgents.B_line import B_lineAgent
    from CybORG.Agents.Wrappers import BlueTableWrapper
    import inspect

    # Load SB3 policy
    sb3 = load_sb3(args.model)

    # Build CybORG env
    path = str(inspect.getfile(CybORG))
    path = path[:-10] + "/Shared/Scenarios/Scenario2.yaml"
    cyborg = CybORG(path, "sim", agents={"Red": B_lineAgent})

    # Precompute Red actions using CybORG's B_line
    red_agent = B_lineAgent()
    cyborg.reset()
    red_actions = precompute_red_sequence(cyborg, red_agent, args.steps)

    # Convert to JAX indices for the JAX env
    red_actions_jax = [cyborg_red_action_to_jax(a, cyborg, "Red") for a in red_actions]

    # Reset both envs for synchronized rollout
    blue_wrapper = BlueTableWrapper(env=cyborg, output_mode="vector")
    blue_wrapper.reset("Blue")
    jax_env = HeuristicRedCAGE()
    key = jax.random.PRNGKey(args.seed)
    obs_jax, state_jax = jax_env.reset(key)

    print("=" * 80)
    print("SB3 policy under scripted Red actions (CybORG vs JAX)")
    print("=" * 80)

    for step in range(args.steps):
        # SB3 action in CybORG
        raw_obs_cyborg = cyborg.get_observation("Blue")
        obs_cyborg = blue_wrapper.observation_change(raw_obs_cyborg)
        blue_action_cyborg, _ = sb3.predict(obs_cyborg, deterministic=True)
        blue_action_cyborg = int(blue_action_cyborg)

        # SB3 action in JAX
        obs_blue_jax = np.array(obs_jax["blue"], dtype=np.float32)
        blue_action_jax, _ = sb3.predict(obs_blue_jax, deterministic=True)
        blue_action_jax = int(blue_action_jax)

        # Apply scripted Red actions
        red_action_cyborg = red_actions[step]
        red_action_jax = red_actions_jax[step]

        # Step CybORG
        blue_action_obj = jax_action_to_cyborg(blue_action_cyborg, cyborg, "Blue")
        cyborg.step("Blue", blue_action_obj)
        cyborg.step("Red", red_action_cyborg)
        raw_obs_cyborg_next = cyborg.get_observation("Blue")
        obs_cyborg_next = blue_wrapper.observation_change(raw_obs_cyborg_next)

        # Step JAX
        key, step_key = jax.random.split(key)
        obs_jax, state_jax, rewards, dones, info = jax_env.step(
            step_key, state_jax, {"blue": blue_action_jax, "red": red_action_jax}
        )

        # Compare observations
        diff = np.abs(np.array(obs_jax["blue"]) - np.array(obs_cyborg_next))
        max_diff = float(np.max(diff))
        print(f"Step {step:2d}: max diff = {max_diff:.3f} | "
              f"Blue(CybORG)={blue_action_cyborg} Blue(JAX)={blue_action_jax}")
        if max_diff > 1e-5:
            print(format_blue_obs_diffs(obs_jax["blue"], obs_cyborg_next))


if __name__ == "__main__":
    main()
