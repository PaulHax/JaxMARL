#!/usr/bin/env python3
"""Evaluate an SB3 PPO policy inside the JAX CAGE environment.

This helps isolate observation/action mismatches between CybORG and JAX envs.
"""
import argparse
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import jax
import jax.numpy as jnp

from jaxmarl.environments.cage import HeuristicRedCAGE


def action_type(action_idx: int, num_hosts: int) -> str:
    from jaxmarl.environments.cage.actions import blue_action_type, BLUE_ACTION_NAMES
    type_idx = blue_action_type(action_idx, num_hosts)
    return BLUE_ACTION_NAMES[type_idx] if type_idx < len(BLUE_ACTION_NAMES) else "Unknown"


def decode_action(action_idx: int, num_hosts: int = 13) -> str:
    from jaxmarl.environments.cage.actions import blue_action_label
    return blue_action_label(action_idx, num_hosts)


def eval_sb3(model_path: str, episodes: int, steps: int, seed: int, deterministic: bool):
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
        import torch
    except ImportError as exc:
        raise SystemExit(f"stable_baselines3/torch required: {exc}")

    model = PPO.load(model_path, device="cpu")

    env = HeuristicRedCAGE()
    action_dim = env.action_space("blue").n
    num_hosts = env.const.num_hosts

    action_counts = np.zeros(action_dim, dtype=np.int64)
    type_counts = {k: 0 for k in ["Sleep", "Monitor", "Analyse", "Remove", "Decoy", "Restore"]}
    topk_counts = np.zeros(action_dim, dtype=np.int64)
    restore_decoy_gaps = []
    decoy_wins = 0

    key = jax.random.PRNGKey(seed)

    for ep in range(episodes):
        key, reset_key = jax.random.split(key)
        obs, state = env.reset(reset_key)
        done = False
        step = 0
        while not done and step < steps:
            obs_blue = np.array(obs["blue"], dtype=np.float32)

            with torch.no_grad():
                obs_tensor = torch.tensor(obs_blue, dtype=torch.float32).unsqueeze(0)
                dist = model.policy.get_distribution(obs_tensor)
                logits = dist.distribution.logits.squeeze(0).cpu().numpy()

            action, _ = model.predict(obs_blue, deterministic=deterministic)
            action = int(action)

            action_counts[action] += 1
            type_counts[action_type(action, num_hosts)] += 1

            # Top-5 logits
            k = min(5, logits.shape[0])
            topk_idx = np.argpartition(-logits, k - 1)[:k]
            for idx in topk_idx:
                topk_counts[idx] += 1

            analyse_start = 2
            remove_start = analyse_start + num_hosts
            decoy_start = remove_start + num_hosts
            restore_start = decoy_start + num_hosts * 8
            restore_end = restore_start + num_hosts
            max_restore = float(np.max(logits[restore_start:restore_end]))
            max_decoy = float(np.max(logits[decoy_start:restore_start]))
            restore_decoy_gaps.append(max_restore - max_decoy)
            if max_decoy > max_restore:
                decoy_wins += 1

            key, step_key = jax.random.split(key)
            obs, state, rewards, dones, info = env.step(step_key, state, {"blue": action})
            done = bool(dones["__all__"])
            step += 1

    total_actions = int(action_counts.sum())
    print("=" * 60)
    mode = "deterministic" if deterministic else "stochastic"
    print(f"SB3 policy in JAX env ({mode})")
    print(f"Episodes: {episodes}, steps/ep: {steps}, total actions: {total_actions}")
    print("=" * 60)

    action_probs = action_counts / max(total_actions, 1)
    sorted_idx = np.argsort(-action_counts)
    print("Top 20 actions by frequency:")
    print("-" * 50)
    for i, idx in enumerate(sorted_idx[:20]):
        if action_counts[idx] > 0:
            print(f"  {i+1:2d}. {decode_action(idx, num_hosts):25s} {action_probs[idx]*100:6.2f}%")

    print()
    print("Action type breakdown:")
    print("-" * 50)
    for k in ["Sleep", "Monitor", "Analyse", "Remove", "Decoy", "Restore"]:
        pct = type_counts[k] / max(total_actions, 1) * 100
        print(f"  {k:8s}: {pct:6.2f}% ({type_counts[k]} actions)")

    print()
    print("Top-5 logits frequency (by action):")
    print("-" * 50)
    topk_probs = topk_counts / max(total_actions, 1)
    topk_sorted = np.argsort(-topk_counts)
    for i, idx in enumerate(topk_sorted[:20]):
        if topk_counts[idx] > 0:
            print(f"  {i+1:2d}. {decode_action(idx, num_hosts):25s} {topk_probs[idx]*100:6.2f}%")

    if restore_decoy_gaps:
        gaps = np.array(restore_decoy_gaps)
        print()
        print("Restore vs Decoy logit gap (max Restore - max Decoy):")
        print("-" * 50)
        print(f"  Mean gap:   {gaps.mean():8.4f}")
        print(f"  Median gap: {np.median(gaps):8.4f}")
        print(f"  P95 gap:    {np.percentile(gaps, 95):8.4f}")
        print(f"  Decoy wins: {decoy_wins/total_actions*100:6.2f}% ({decoy_wins} steps)")


def main():
    parser = argparse.ArgumentParser(description="Evaluate SB3 policy in JAX CAGE environment")
    parser.add_argument("--model", required=True, help="Path to SB3 .zip model")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=["deterministic", "stochastic", "both"], default="both")
    args = parser.parse_args()

    if args.mode in ("deterministic", "both"):
        eval_sb3(args.model, args.episodes, args.steps, args.seed, deterministic=True)
    if args.mode in ("stochastic", "both"):
        eval_sb3(args.model, args.episodes, args.steps, args.seed, deterministic=False)


if __name__ == "__main__":
    main()
