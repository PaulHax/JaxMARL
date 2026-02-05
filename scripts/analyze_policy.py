#!/usr/bin/env python
"""Analyze action distribution of trained CAGE policy."""
import argparse
import pickle
import sys
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import distrax
from flax import linen as nn
from flax.linen.initializers import constant, orthogonal

sys.path.insert(0, str(Path(__file__).parent.parent))
from jaxmarl.environments.cage import HeuristicRedCAGE
from jaxmarl.environments.cage.actions import get_blue_action_offsets

class ActorCriticSeparate(nn.Module):
    action_dim: int
    hidden_dim: int = 64  # Will be overridden based on checkpoint

    @nn.compact
    def __call__(self, x):
        actor = nn.Dense(self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        actor = nn.tanh(actor)
        actor = nn.Dense(self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(actor)
        actor = nn.tanh(actor)
        actor = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor)
        pi = distrax.Categorical(logits=actor)

        critic = nn.Dense(self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        critic = nn.tanh(critic)
        critic = nn.Dense(self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(critic)
        critic = nn.tanh(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)
        return pi, jnp.squeeze(critic, axis=-1)


class ActorCriticShared(nn.Module):
    action_dim: int
    hidden_dim: int = 64

    @nn.compact
    def __call__(self, x):
        trunk = nn.Dense(self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        trunk = nn.tanh(trunk)
        trunk = nn.Dense(self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(trunk)
        trunk = nn.tanh(trunk)

        logits = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(trunk)
        pi = distrax.Categorical(logits=logits)

        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(trunk)
        return pi, jnp.squeeze(critic, axis=-1)


def infer_network_config(params):
    if isinstance(params, dict) and 'params' in params:
        params = params['params']
    num_dense = sum(1 for k in params.keys() if k.startswith('Dense_'))
    hidden_dim = params['Dense_0']['bias'].shape[0]
    network_type = "shared" if num_dense == 4 else "separate"
    return network_type, int(hidden_dim)

def get_action_name(action_idx, num_hosts=13):
    if action_idx == 0:
        return "Sleep"
    elif action_idx == 1:
        return "Monitor"
    base = 2
    if action_idx < base + num_hosts:
        return f"Analyse({action_idx - base})"
    base += num_hosts
    if action_idx < base + num_hosts:
        return f"Remove({action_idx - base})"
    base += num_hosts
    num_decoys = 8  # 8 decoy types in CAGE, not 10
    total_decoy = num_hosts * num_decoys
    if action_idx < base + total_decoy:
        rel = action_idx - base
        host = rel // num_decoys
        decoy = rel % num_decoys
        return f"Decoy({host},{decoy})"
    base += total_decoy
    if action_idx < base + num_hosts:
        return f"Restore({action_idx - base})"
    return f"Unknown({action_idx})"

def _action_type(action_idx, num_hosts=13):
    if action_idx == 0:
        return "Sleep"
    elif action_idx == 1:
        return "Monitor"
    base = 2
    if action_idx < base + num_hosts:
        return "Analyse"
    base += num_hosts
    if action_idx < base + num_hosts:
        return "Remove"
    base += num_hosts
    num_decoys = 8
    total_decoy = num_hosts * num_decoys
    if action_idx < base + total_decoy:
        return "Decoy"
    base += total_decoy
    if action_idx < base + num_hosts:
        return "Restore"
    return "Unknown"


def analyze_policy(checkpoint_path: str, num_episodes: int = 10, top_k: int = 5):
    with open(checkpoint_path, 'rb') as f:
        ckpt = pickle.load(f)

    env = HeuristicRedCAGE(scenario="Scenario2")
    obs_dim = env.observation_spaces["blue"].shape[0]
    action_dim = env.action_spaces["blue"].n

    # Detect network type and hidden dim from checkpoint
    params = ckpt['params'] if 'params' in ckpt else ckpt['runner_state'][0].params
    network_type, hidden_dim = infer_network_config(params)
    print(f"Detected network_type={network_type}, hidden_dim={hidden_dim} from checkpoint")

    if network_type == "shared":
        network = ActorCriticShared(action_dim=action_dim, hidden_dim=hidden_dim)
    else:
        network = ActorCriticSeparate(action_dim=action_dim, hidden_dim=hidden_dim)

    action_counts = np.zeros(action_dim)
    topk_counts = np.zeros(action_dim)
    argmax_type_counts = {k: 0 for k in ["Sleep", "Monitor", "Analyse", "Remove", "Decoy", "Restore", "Unknown"]}
    restore_decoy_gaps = []
    decoy_wins = 0
    total_actions = 0
    episode_rewards = []

    key = jax.random.PRNGKey(42)

    for ep in range(num_episodes):
        key, reset_key = jax.random.split(key)
        obs, state = env.reset(reset_key)
        done = False
        ep_reward = 0.0
        step = 0

        while not done and step < 100:
            obs_blue = obs["blue"]
            pi, _ = network.apply(params, obs_blue)
            logits = np.array(pi.logits)
            key, action_key = jax.random.split(key)
            action = pi.sample(seed=action_key)
            action_int = int(action)
            action_counts[action_int] += 1
            total_actions += 1

            # Argmax diagnostics
            argmax_action = int(np.argmax(logits))
            argmax_type_counts[_action_type(argmax_action, num_hosts=env.const.num_hosts)] += 1

            # Top-K logits frequency
            if top_k > 0:
                k = min(top_k, logits.shape[0])
                topk_idx = np.argpartition(-logits, k - 1)[:k]
                for idx in topk_idx:
                    topk_counts[idx] += 1

            # Restore vs Decoy logit gap
            analyse_start, remove_start, decoy_start, restore_start = get_blue_action_offsets(env.const)
            restore_end = restore_start + env.const.num_hosts
            decoy_end = restore_start
            max_restore = float(np.max(logits[restore_start:restore_end]))
            max_decoy = float(np.max(logits[decoy_start:decoy_end]))
            restore_decoy_gaps.append(max_restore - max_decoy)
            if max_decoy > max_restore:
                decoy_wins += 1

            key, step_key = jax.random.split(key)
            actions = {"blue": action, "red": jnp.array(0)}
            obs, state, reward, done_dict, info = env.step(step_key, state, actions)
            done = done_dict["__all__"]
            ep_reward += float(reward["blue"])
            step += 1

        episode_rewards.append(ep_reward)

    print("=" * 60)
    print("POLICY ACTION DISTRIBUTION ANALYSIS")
    print("=" * 60)
    print(f"Analyzed {num_episodes} episodes, {total_actions} total actions")
    print(f"Mean episode reward: {np.mean(episode_rewards):.2f} (+/- {np.std(episode_rewards):.2f})")
    print()

    action_probs = action_counts / total_actions
    sorted_indices = np.argsort(-action_counts)

    print("Top 20 actions by frequency (sampled actions):")
    print("-" * 50)
    for i, idx in enumerate(sorted_indices[:20]):
        if action_counts[idx] > 0:
            name = get_action_name(idx)
            print(f"  {i+1:2d}. {name:25s} {action_probs[idx]*100:6.2f}% ({int(action_counts[idx])} times)")

    print()
    print("Action type breakdown:")
    print("-" * 50)
    sleep_count = action_counts[0]
    monitor_count = action_counts[1]
    analyse_count = sum(action_counts[2:15])
    remove_count = sum(action_counts[15:28])
    decoy_count = sum(action_counts[28:132])  # 13 hosts * 8 decoys = 104 actions
    restore_count = sum(action_counts[132:145])  # 13 hosts

    print(f"  Sleep:    {sleep_count/total_actions*100:6.2f}% ({int(sleep_count)} actions)")
    print(f"  Monitor:  {monitor_count/total_actions*100:6.2f}% ({int(monitor_count)} actions)")
    print(f"  Analyse:  {analyse_count/total_actions*100:6.2f}% ({int(analyse_count)} actions)")
    print(f"  Remove:   {remove_count/total_actions*100:6.2f}% ({int(remove_count)} actions)")
    print(f"  Decoy:    {decoy_count/total_actions*100:6.2f}% ({int(decoy_count)} actions)")
    print(f"  Restore:  {restore_count/total_actions*100:6.2f}% ({int(restore_count)} actions)")
    print()

    print("Argmax action type breakdown (logits):")
    print("-" * 50)
    for k, v in argmax_type_counts.items():
        print(f"  {k:8s}: {v/total_actions*100:6.2f}% ({int(v)} steps)")
    print()

    if top_k > 0:
        print(f"Top {top_k} logits frequency (by action):")
        print("-" * 50)
        topk_probs = topk_counts / total_actions
        topk_sorted = np.argsort(-topk_counts)
        for i, idx in enumerate(topk_sorted[:20]):
            if topk_counts[idx] > 0:
                name = get_action_name(idx)
                print(f"  {i+1:2d}. {name:25s} {topk_probs[idx]*100:6.2f}% ({int(topk_counts[idx])} steps)")
        print()

    if restore_decoy_gaps:
        gaps = np.array(restore_decoy_gaps)
        print("Restore vs Decoy logit gap (max Restore - max Decoy):")
        print("-" * 50)
        print(f"  Mean gap:   {gaps.mean():8.4f}")
        print(f"  Median gap: {np.median(gaps):8.4f}")
        print(f"  P95 gap:    {np.percentile(gaps, 95):8.4f}")
        print(f"  Decoy wins: {decoy_wins/total_actions*100:6.2f}% ({decoy_wins} steps)")
        print()

    print("Entropy analysis:")
    print("-" * 50)
    nonzero_probs = action_probs[action_probs > 0]
    entropy = -np.sum(nonzero_probs * np.log(nonzero_probs))
    max_entropy = np.log(action_dim)
    print(f"  Policy entropy:    {entropy:.3f}")
    print(f"  Max entropy:       {max_entropy:.3f}")
    print(f"  Normalized:        {entropy/max_entropy*100:.1f}%")
    print(f"  Unique actions:    {np.sum(action_counts > 0)} / {action_dim}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze action distribution and logits of CAGE policy")
    parser.add_argument("checkpoint", nargs="?", default="experiments/2026-01-27_210130_blue-vs-bline_seed0/checkpoint_final.pkl")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()
    analyze_policy(args.checkpoint, num_episodes=args.episodes, top_k=args.top_k)
