#!/usr/bin/env python
"""Training script for CAGE-JAX environment.

Example usage:
    cd JaxMARL && python scripts/train_cage.py
    python scripts/train_cage.py --num_envs 1024 --num_steps 1000000
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for running from repo root
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))
import time

import jax
import jax.numpy as jnp

from jaxmarl.environments.cage import CageEnv


def parse_args():
    parser = argparse.ArgumentParser(description="Train agents on CAGE-JAX")
    parser.add_argument("--num_envs", type=int, default=64, help="Number of parallel environments")
    parser.add_argument("--num_steps", type=int, default=100000, help="Total training steps")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--eval_interval", type=int, default=10000, help="Steps between evaluations")
    return parser.parse_args()


def random_policy(key, avail_actions):
    """Sample random valid action."""
    probs = avail_actions.astype(jnp.float32)
    probs = probs / jnp.sum(probs)
    return jax.random.choice(key, jnp.arange(len(avail_actions)), p=probs)


def run_episode(env, key):
    """Run single episode with random policy, return total rewards."""
    obs, state = env.reset(key)
    total_blue = 0.0
    total_red = 0.0

    for _ in range(100):
        key, key_blue, key_red, key_step = jax.random.split(key, 4)

        avail = env.get_avail_actions(state)
        blue_action = random_policy(key_blue, avail["blue"])
        red_action = random_policy(key_red, avail["red"])

        actions = {"blue": blue_action, "red": red_action}
        obs, state, rewards, dones, info = env.step(key_step, state, actions)

        total_blue += rewards["blue"]
        total_red += rewards["red"]

        if dones["__all__"]:
            break

    return total_blue, total_red


def evaluate(env, key, num_episodes=100):
    """Evaluate with random policy."""
    blue_rewards = []
    red_rewards = []

    for i in range(num_episodes):
        key, episode_key = jax.random.split(key)
        blue_r, red_r = run_episode(env, episode_key)
        blue_rewards.append(float(blue_r))
        red_rewards.append(float(red_r))

    return {
        "blue_mean": jnp.mean(jnp.array(blue_rewards)),
        "blue_std": jnp.std(jnp.array(blue_rewards)),
        "red_mean": jnp.mean(jnp.array(red_rewards)),
        "red_std": jnp.std(jnp.array(red_rewards)),
    }


def main():
    args = parse_args()

    print("=" * 50)
    print("CAGE-JAX Training")
    print("=" * 50)
    print(f"JAX devices: {jax.devices()}")
    print(f"Num environments: {args.num_envs}")
    print(f"Total steps: {args.num_steps}")
    print("=" * 50)

    env = CageEnv()
    key = jax.random.PRNGKey(args.seed)

    # Initial evaluation
    print("\nInitial evaluation (random policy)...")
    key, eval_key = jax.random.split(key)
    start = time.perf_counter()
    metrics = evaluate(env, eval_key, num_episodes=100)
    eval_time = time.perf_counter() - start

    print(f"Blue reward: {metrics['blue_mean']:.2f} ± {metrics['blue_std']:.2f}")
    print(f"Red reward: {metrics['red_mean']:.2f} ± {metrics['red_std']:.2f}")
    print(f"Evaluation time: {eval_time:.2f}s")

    # Measure throughput
    print("\nMeasuring throughput...")
    from jaxmarl.environments.cage.actions import NUM_BLUE_ACTIONS, NUM_RED_ACTIONS

    # JIT compile
    step_jit = jax.jit(env.step)
    reset_jit = jax.jit(env.reset)

    key, reset_key = jax.random.split(key)
    obs, state = reset_jit(reset_key)

    # Warmup
    actions = {"blue": jnp.array(0), "red": jnp.array(0)}
    for _ in range(100):
        key, step_key = jax.random.split(key)
        obs, state, rewards, dones, info = step_jit(step_key, state, actions)
    jax.block_until_ready(state)

    # Timed run
    num_bench_steps = 10000
    start = time.perf_counter()
    for _ in range(num_bench_steps):
        key, step_key = jax.random.split(key)
        obs, state, rewards, dones, info = step_jit(step_key, state, actions)
    jax.block_until_ready(state)
    elapsed = time.perf_counter() - start

    sps = num_bench_steps / elapsed
    print(f"Single env: {sps:.1f} steps/sec")

    print("\n✓ Training script complete")
    print("For full IPPO training, integrate with JaxMARL baselines.")


if __name__ == "__main__":
    main()
