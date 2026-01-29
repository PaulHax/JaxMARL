#!/usr/bin/env python3
"""Debug policy divergence between JAX and CybORG environments.

Runs a trained JAX policy step-by-step in both environments,
comparing observations, actions, and rewards to find where they diverge.
"""

import argparse
import inspect
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

# CybORG imports
from CybORG import CybORG
from CybORG.Agents import B_lineAgent
from CybORG.Agents.Wrappers import ChallengeWrapper

# JAX environment imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from jaxmarl.environments.cage import HeuristicRedCAGE
from baselines.IPPO.ippo_ff_cage import ActorCritic  # Network architecture


def load_jax_policy(checkpoint_path: str):
    """Load trained JAX policy from checkpoint."""
    with open(checkpoint_path, 'rb') as f:
        data = pickle.load(f)

    # Handle different checkpoint formats
    # ippo_ff_cage saves as {'params': {'params': actual_params}}
    if 'params' in data:
        params = data['params']
        # Check for nested params
        if isinstance(params, dict) and 'params' in params:
            params = params
        return params
    elif 'actor_params' in data:
        return data['actor_params']
    else:
        return data


def create_jax_env(seed: int = 42):
    """Create JAX HeuristicRedCAGE environment."""
    env = HeuristicRedCAGE()
    return env


def create_cyborg_env():
    """Create CybORG environment with B_lineAgent."""
    path = str(inspect.getfile(CybORG))
    path = path[:-10] + '/Shared/Scenarios/Scenario2.yaml'
    cyborg = CybORG(path, 'sim', agents={'Red': B_lineAgent})
    wrapped = ChallengeWrapper(env=cyborg, agent_name='Blue')
    return wrapped, cyborg


def get_jax_action(network, params, obs, key):
    """Get action from JAX policy."""
    obs_batch = jnp.expand_dims(obs, 0)  # Add batch dim
    pi, _ = network.apply(params, obs_batch)

    # Get action probabilities
    probs = jax.nn.softmax(pi.logits[0])

    # Sample action
    action = jax.random.categorical(key, pi.logits[0])

    return int(action), probs


def get_jax_action_deterministic(network, params, obs):
    """Get deterministic action from JAX policy (argmax)."""
    obs_batch = jnp.expand_dims(obs, 0)
    pi, _ = network.apply(params, obs_batch)
    probs = jax.nn.softmax(pi.logits[0])
    action = jnp.argmax(pi.logits[0])
    return int(action), probs


def compare_observations(jax_obs, cyborg_obs, step: int, tolerance: float = 1e-5):
    """Compare observations between JAX and CybORG."""
    jax_obs_np = np.array(jax_obs)
    cyborg_obs_np = np.array(cyborg_obs)

    if jax_obs_np.shape != cyborg_obs_np.shape:
        return False, f"Shape mismatch: JAX {jax_obs_np.shape} vs CybORG {cyborg_obs_np.shape}"

    diff = np.abs(jax_obs_np - cyborg_obs_np)
    max_diff = np.max(diff)

    if max_diff > tolerance:
        diff_indices = np.where(diff > tolerance)[0]
        return False, f"Max diff: {max_diff:.6f} at indices {diff_indices[:10]}"

    return True, f"Match (max diff: {max_diff:.6f})"


def decode_action(action_idx: int, num_hosts: int = 13) -> str:
    """Decode action index to human-readable string.

    CybORG action layout (matching JAX):
    - 0: Sleep
    - 1: Monitor
    - 2-14: Analyse (13 hosts)
    - 15-27: Remove (13 hosts)
    - 28-131: Decoy (8 types × 13 hosts, decoy_type first then host)
    - 132-144: Restore (13 hosts)

    Host order (alphabetical): Defender, Enterprise0-2, Op_Host0-2, Op_Server0, User0-4
    Decoy order: Apache, Femitter, HarakaSMPT, Smss, SSHD, Svchost, Tomcat, Vsftpd
    """
    host_names = [
        "Defender", "Enterprise0", "Enterprise1", "Enterprise2",
        "Op_Host0", "Op_Host1", "Op_Host2", "Op_Server0",
        "User0", "User1", "User2", "User3", "User4"
    ]
    decoy_names = ["Apache", "Femitter", "HarakaSMPT", "Smss",
                   "SSHD", "Svchost", "Tomcat", "Vsftpd"]
    num_decoys = 8

    if action_idx == 0:
        return "Sleep"
    elif action_idx == 1:
        return "Monitor"
    elif action_idx < 2 + num_hosts:  # 2-14
        return f"Analyse({host_names[action_idx - 2]})"
    elif action_idx < 2 + 2 * num_hosts:  # 15-27
        return f"Remove({host_names[action_idx - 2 - num_hosts]})"
    elif action_idx < 2 + 2 * num_hosts + num_decoys * num_hosts:  # 28-131
        decoy_offset = action_idx - 2 - 2 * num_hosts
        decoy_type = decoy_offset // num_hosts
        host_idx = decoy_offset % num_hosts
        return f"Decoy{decoy_names[decoy_type]}({host_names[host_idx]})"
    else:  # 132-144
        return f"Restore({host_names[action_idx - 2 - 2 * num_hosts - num_decoys * num_hosts]})"


def run_comparison(checkpoint_path: str, num_steps: int = 100, seed: int = 42,
                   deterministic: bool = True, verbose: bool = True):
    """Run policy in both environments and compare."""

    print(f"Loading checkpoint: {checkpoint_path}")
    params = load_jax_policy(checkpoint_path)

    # Create network
    jax_env = create_jax_env(seed)
    obs_shape = jax_env.observation_space('blue').shape[0]
    action_dim = jax_env.action_space('blue').n

    network = ActorCritic(action_dim=action_dim, activation="tanh")

    # Initialize environments
    key = jax.random.PRNGKey(seed)
    key, reset_key = jax.random.split(key)

    jax_obs, jax_state = jax_env.reset(reset_key)
    jax_obs = jax_obs['blue']

    cyborg_wrapped, cyborg_raw = create_cyborg_env()
    cyborg_obs = cyborg_wrapped.reset()
    if isinstance(cyborg_obs, tuple):
        cyborg_obs = cyborg_obs[0]

    print(f"\n{'='*80}")
    print(f"Starting comparison: {num_steps} steps, seed={seed}, deterministic={deterministic}")
    print(f"JAX obs shape: {jax_obs.shape}, CybORG obs shape: {np.array(cyborg_obs).shape}")
    print(f"Action space: {action_dim}")
    print(f"{'='*80}\n")

    jax_total_reward = 0
    cyborg_total_reward = 0
    divergences = []

    for step in range(num_steps):
        key, action_key = jax.random.split(key)

        # Compare observations
        obs_match, obs_msg = compare_observations(jax_obs, cyborg_obs, step)

        # Get action from policy
        if deterministic:
            action, probs = get_jax_action_deterministic(network, params, jax_obs)
        else:
            action, probs = get_jax_action(network, params, jax_obs, action_key)

        action_name = decode_action(action)

        # Step both environments
        key, step_key = jax.random.split(key)
        jax_obs_next, jax_state, jax_reward_dict, jax_done, jax_info = jax_env.step(
            step_key, jax_state, {'blue': action}
        )
        jax_obs_next = jax_obs_next['blue']
        jax_reward = float(jax_reward_dict['blue'])

        cyborg_result = cyborg_wrapped.step(action)
        if len(cyborg_result) == 5:
            cyborg_obs_next, cyborg_reward, cyborg_term, cyborg_trunc, cyborg_info = cyborg_result
        else:
            cyborg_obs_next, cyborg_reward, cyborg_done, cyborg_info = cyborg_result

        cyborg_action_str = str(cyborg_raw.get_last_action('Blue'))
        cyborg_red_action = str(cyborg_raw.get_last_action('Red'))

        jax_total_reward += jax_reward
        cyborg_total_reward += cyborg_reward

        # Check for divergence
        reward_match = abs(jax_reward - cyborg_reward) < 0.01

        if verbose or not obs_match or not reward_match:
            print(f"Step {step:3d}: action={action:3d} ({action_name})")
            print(f"         JAX reward: {jax_reward:8.2f}, CybORG reward: {cyborg_reward:8.2f}")
            print(f"         Obs: {obs_msg}")
            print(f"         CybORG Blue: {cyborg_action_str}")
            print(f"         CybORG Red:  {cyborg_red_action}")

            if not obs_match:
                print(f"         *** OBSERVATION DIVERGENCE ***")
                divergences.append(('obs', step, obs_msg))
            if not reward_match:
                print(f"         *** REWARD DIVERGENCE: JAX={jax_reward:.2f} vs CybORG={cyborg_reward:.2f} ***")
                divergences.append(('reward', step, f"JAX={jax_reward:.2f} vs CybORG={cyborg_reward:.2f}"))
            print()

        # Update observations
        jax_obs = jax_obs_next
        cyborg_obs = cyborg_obs_next

    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"JAX total reward:    {jax_total_reward:8.2f}")
    print(f"CybORG total reward: {cyborg_total_reward:8.2f}")
    print(f"Difference:          {jax_total_reward - cyborg_total_reward:8.2f}")
    print(f"\nDivergences found: {len(divergences)}")

    if divergences:
        print("\nFirst 10 divergences:")
        for dtype, step, msg in divergences[:10]:
            print(f"  Step {step}: {dtype} - {msg}")

    return divergences


def main():
    parser = argparse.ArgumentParser(description="Debug policy divergence between JAX and CybORG")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to JAX policy checkpoint")
    parser.add_argument("--steps", type=int, default=100,
                        help="Number of steps to run")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--stochastic", action="store_true",
                        help="Use stochastic policy (default: deterministic)")
    parser.add_argument("--quiet", action="store_true",
                        help="Only print divergences")

    args = parser.parse_args()

    run_comparison(
        checkpoint_path=args.checkpoint,
        num_steps=args.steps,
        seed=args.seed,
        deterministic=not args.stochastic,
        verbose=not args.quiet
    )


if __name__ == "__main__":
    main()
