#!/usr/bin/env python
"""Benchmark script for CAGE-JAX performance.

Measures:
- Single environment steps/sec (JIT speedup)
- Parallel environment steps/sec (vmap speedup)
- Memory usage

Example usage:
    cd JaxMARL && python benchmarks/cage_benchmark.py
    python benchmarks/cage_benchmark.py --num_envs 4096
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for running from repo root
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))
import json
import platform
import time
from datetime import datetime

import jax
import jax.numpy as jnp

from jaxmarl.environments.cage import CageEnv


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark CAGE-JAX")
    parser.add_argument("--num_envs", type=int, default=1024, help="Number of parallel environments")
    parser.add_argument("--num_steps", type=int, default=10000, help="Steps per benchmark")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Warmup steps")
    parser.add_argument("--output", type=str, default="benchmark_results.json", help="Output file")
    return parser.parse_args()


def get_hardware_info():
    """Collect hardware information."""
    info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "jax_version": jax.__version__,
        "jax_devices": [str(d) for d in jax.devices()],
    }
    return info


def benchmark_single_env(num_steps=10000, warmup_steps=100):
    """Benchmark single environment throughput."""
    env = CageEnv()

    step_jit = jax.jit(env.step)
    reset_jit = jax.jit(env.reset)

    key = jax.random.PRNGKey(0)
    key, reset_key = jax.random.split(key)
    obs, state = reset_jit(reset_key)

    actions = {"blue": jnp.array(0), "red": jnp.array(0)}

    # Warmup
    for _ in range(warmup_steps):
        key, step_key = jax.random.split(key)
        obs, state, rewards, dones, info = step_jit(step_key, state, actions)
    jax.block_until_ready(state)

    # Timed run
    start = time.perf_counter()
    for _ in range(num_steps):
        key, step_key = jax.random.split(key)
        obs, state, rewards, dones, info = step_jit(step_key, state, actions)
        if dones["__all__"]:
            key, reset_key = jax.random.split(key)
            obs, state = reset_jit(reset_key)
    jax.block_until_ready(state)
    elapsed = time.perf_counter() - start

    return {
        "steps": num_steps,
        "time_seconds": elapsed,
        "steps_per_second": num_steps / elapsed,
    }


def benchmark_parallel_envs(num_envs=1024, num_steps=10000, warmup_steps=100):
    """Benchmark parallel environment throughput."""
    env = CageEnv()

    # Vectorize functions
    reset_vmap = jax.vmap(env.reset)
    step_vmap = jax.vmap(env.step)

    reset_jit = jax.jit(reset_vmap)
    step_jit = jax.jit(step_vmap)

    # Initialize
    keys = jax.random.split(jax.random.PRNGKey(0), num_envs)
    obs, states = reset_jit(keys)

    actions = {
        "blue": jnp.zeros(num_envs, dtype=jnp.int32),
        "red": jnp.zeros(num_envs, dtype=jnp.int32),
    }

    # Warmup
    for _ in range(warmup_steps):
        keys = jax.random.split(keys[0], num_envs)
        obs, states, rewards, dones, info = step_jit(keys, states, actions)
    jax.block_until_ready(states)

    # Timed run
    start = time.perf_counter()
    for _ in range(num_steps):
        keys = jax.random.split(keys[0], num_envs)
        obs, states, rewards, dones, info = step_jit(keys, states, actions)
    jax.block_until_ready(states)
    elapsed = time.perf_counter() - start

    total_steps = num_steps * num_envs
    return {
        "num_envs": num_envs,
        "steps_per_env": num_steps,
        "total_steps": total_steps,
        "time_seconds": elapsed,
        "steps_per_second": total_steps / elapsed,
        "env_steps_per_second": num_steps / elapsed,
    }


def benchmark_episode_rollout(num_episodes=100):
    """Benchmark full episode rollouts."""
    env = CageEnv()

    step_jit = jax.jit(env.step)
    reset_jit = jax.jit(env.reset)

    key = jax.random.PRNGKey(42)
    total_steps = 0

    start = time.perf_counter()
    for ep in range(num_episodes):
        key, reset_key = jax.random.split(key)
        obs, state = reset_jit(reset_key)

        actions = {"blue": jnp.array(0), "red": jnp.array(0)}
        for step in range(100):
            key, step_key = jax.random.split(key)
            obs, state, rewards, dones, info = step_jit(step_key, state, actions)
            total_steps += 1
            if dones["__all__"]:
                break
    jax.block_until_ready(state)
    elapsed = time.perf_counter() - start

    return {
        "num_episodes": num_episodes,
        "total_steps": total_steps,
        "time_seconds": elapsed,
        "episodes_per_second": num_episodes / elapsed,
        "steps_per_second": total_steps / elapsed,
    }


def main():
    args = parse_args()

    print("=" * 60)
    print("CAGE-JAX Benchmark")
    print("=" * 60)

    hardware = get_hardware_info()
    print(f"Platform: {hardware['platform']}")
    print(f"JAX version: {hardware['jax_version']}")
    print(f"JAX devices: {hardware['jax_devices']}")
    print("=" * 60)

    results = {
        "timestamp": datetime.now().isoformat(),
        "hardware": hardware,
    }

    # Single environment benchmark
    print("\n[1/3] Benchmarking single environment...")
    single_results = benchmark_single_env(
        num_steps=args.num_steps,
        warmup_steps=args.warmup_steps,
    )
    results["single_env"] = single_results
    print(f"  Steps: {single_results['steps']}")
    print(f"  Time: {single_results['time_seconds']:.2f}s")
    print(f"  Throughput: {single_results['steps_per_second']:.1f} steps/sec")

    # Parallel environment benchmark
    print(f"\n[2/3] Benchmarking {args.num_envs} parallel environments...")
    parallel_results = benchmark_parallel_envs(
        num_envs=args.num_envs,
        num_steps=args.num_steps // 10,  # Fewer steps per env
        warmup_steps=args.warmup_steps // 10,
    )
    results["parallel_envs"] = parallel_results
    print(f"  Total steps: {parallel_results['total_steps']}")
    print(f"  Time: {parallel_results['time_seconds']:.2f}s")
    print(f"  Throughput: {parallel_results['steps_per_second']:.1f} steps/sec")
    print(f"  Speedup: {parallel_results['steps_per_second'] / single_results['steps_per_second']:.1f}x")

    # Episode rollout benchmark
    print("\n[3/3] Benchmarking episode rollouts...")
    episode_results = benchmark_episode_rollout(num_episodes=100)
    results["episode_rollout"] = episode_results
    print(f"  Episodes: {episode_results['num_episodes']}")
    print(f"  Time: {episode_results['time_seconds']:.2f}s")
    print(f"  Episodes/sec: {episode_results['episodes_per_second']:.1f}")

    # Calculate speedup metrics
    results["speedup"] = {
        "parallel_vs_single": parallel_results['steps_per_second'] / single_results['steps_per_second'],
    }

    # Summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"Single env:     {single_results['steps_per_second']:>12.1f} steps/sec")
    print(f"Parallel envs:  {parallel_results['steps_per_second']:>12.1f} steps/sec ({args.num_envs} envs)")
    print(f"Parallel speedup: {results['speedup']['parallel_vs_single']:.1f}x")
    print("=" * 60)

    # Save results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
