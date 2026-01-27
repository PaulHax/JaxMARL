#!/usr/bin/env python
"""Train Blue agent using PPO against fixed B_lineAgent Red.

This script trains a Blue agent against the deterministic B_lineAgent,
matching the training setup from the cage-2 fork for reproducibility.

Each run creates an experiment directory with:
  - config.json: All hyperparameters and settings
  - metrics.jsonl: Per-update training metrics
  - checkpoint_final.pkl: Trained model weights
  - reproduce.sh: Script to reproduce the experiment
  - environment.txt: JAX version and device info

Usage:
    python scripts/train_cage_vs_bline.py
    python scripts/train_cage_vs_bline.py --seed 42 --total_timesteps 1000000

View experiments:
    mlflow ui  # then open http://localhost:5000
"""

import argparse
import json
import subprocess
import time
import pickle
from datetime import datetime
from functools import partial
from pathlib import Path

import jax
import mlflow
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from flax.training.train_state import TrainState

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from jaxmarl.environments.cage import CageEnv
from jaxmarl.environments.cage.actions import NUM_BLUE_ACTIONS
from jaxmarl.environments.cage.observations import BLUE_OBS_DIM
from jaxmarl.environments.cage.scripted_agents import (
    BLineState,
    bline_reset_batched,
    bline_get_action_batched,
)


class ActorCritic(nn.Module):
    """Simple actor-critic network."""
    action_dim: int
    hidden_dim: int = 256
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        act_fn = nn.tanh if self.activation == "tanh" else nn.relu
        x = nn.Dense(self.hidden_dim)(x)
        x = act_fn(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = act_fn(x)
        logits = nn.Dense(self.action_dim)(x)
        value = nn.Dense(1)(x)
        return logits, value.squeeze(-1)


def create_train_state(key, obs_dim, action_dim, learning_rate=3e-4, hidden_dim=256, activation="tanh", max_grad_norm=0.5):
    """Create training state for an agent."""
    network = ActorCritic(action_dim=action_dim, hidden_dim=hidden_dim, activation=activation)
    dummy_obs = jnp.zeros((1, obs_dim))
    params = network.init(key, dummy_obs)
    if max_grad_norm > 0:
        tx = optax.chain(
            optax.clip_by_global_norm(max_grad_norm),
            optax.adam(learning_rate),
        )
    else:
        tx = optax.adam(learning_rate)
    return TrainState.create(apply_fn=network.apply, params=params, tx=tx)


def sample_action(key, logits, avail_mask):
    """Sample action from policy logits with action masking."""
    masked_logits = jnp.where(avail_mask, logits, -1e10)
    return jax.random.categorical(key, masked_logits)


@partial(jax.jit, static_argnums=[1, 5])
def collect_rollout(key, env, states, train_state_blue, bline_states, num_steps=128):
    """Collect rollout data from parallel environments.

    Uses learned Blue policy and fixed B_lineAgent for Red.
    """

    def step_fn(carry, _):
        key, env_states, obs, bline_states = carry

        key, key_blue, key_red, key_step = jax.random.split(key, 4)

        # Blue: learned policy
        blue_logits, blue_values = train_state_blue.apply_fn(
            train_state_blue.params, obs['blue']
        )

        avail = jax.vmap(env.get_avail_actions)(env_states)

        blue_actions = jax.vmap(sample_action)(
            jax.random.split(key_blue, env_states.time.shape[0]),
            blue_logits,
            avail['blue']
        )

        # Red: fixed B_lineAgent
        red_keys = jax.random.split(key_red, env_states.time.shape[0])
        red_actions, new_bline_states = bline_get_action_batched(
            bline_states, obs['red'], avail['red'], env.const, red_keys
        )

        actions = {'blue': blue_actions, 'red': red_actions}

        keys_step = jax.random.split(key_step, env_states.time.shape[0])
        next_obs, next_states, rewards, dones, infos = jax.vmap(env.step)(
            keys_step, env_states, actions
        )

        # Reset B_lineAgent state when episode ends
        batch_size = env_states.time.shape[0]
        reset_bline = bline_reset_batched(batch_size)
        new_bline_states = BLineState(
            fsm_state=jnp.where(dones['__all__'], reset_bline.fsm_state, new_bline_states.fsm_state),
            last_action_success=jnp.where(dones['__all__'], reset_bline.last_action_success, new_bline_states.last_action_success),
        )

        transition = {
            'obs_blue': obs['blue'],
            'action_blue': blue_actions,
            'reward_blue': rewards['blue'],
            'done': dones['__all__'],
            'value_blue': blue_values,
            'logits_blue': blue_logits,
            'avail_blue': avail['blue'],
        }

        return (key, next_states, next_obs, new_bline_states), transition

    obs = jax.vmap(env.get_obs)(states)

    (key, final_states, final_obs, final_bline_states), transitions = jax.lax.scan(
        step_fn, (key, states, obs, bline_states), None, length=num_steps
    )

    return key, final_states, final_obs, final_bline_states, transitions


@jax.jit
def compute_gae(rewards, values, dones, gamma=0.99, gae_lambda=0.95):
    """Compute Generalized Advantage Estimation."""
    next_values = jnp.concatenate([values[1:], jnp.zeros_like(values[:1])])
    deltas = rewards + gamma * next_values * (1 - dones) - values

    def scan_fn(lastgae, inputs):
        delta, done = inputs
        advantage = delta + gamma * gae_lambda * (1 - done) * lastgae
        return advantage, advantage

    _, advantages_reversed = jax.lax.scan(
        scan_fn,
        jnp.zeros(rewards.shape[1:]),
        (deltas[::-1], dones[::-1])
    )
    advantages = advantages_reversed[::-1]
    returns = advantages + values
    return advantages, returns


def ppo_loss(params, apply_fn, obs, actions, old_logits, advantages, returns, avail_mask=None, clip_eps=0.2, vf_coef=0.5, ent_coef=0.0):
    """PPO clipped objective loss with proper action masking.

    Follows SB3's MaskablePPO approach: masked actions don't contribute to entropy.
    """
    logits, values = apply_fn(params, obs)

    # Apply action masking to logits for probability computation
    if avail_mask is not None:
        masked_logits = jnp.where(avail_mask, logits, -1e10)
        masked_old_logits = jnp.where(avail_mask, old_logits, -1e10)
    else:
        masked_logits = logits
        masked_old_logits = old_logits

    log_probs = jax.nn.log_softmax(masked_logits)
    old_log_probs = jax.nn.log_softmax(masked_old_logits)

    action_log_probs = jnp.take_along_axis(log_probs, actions[:, None], axis=1).squeeze()
    old_action_log_probs = jnp.take_along_axis(old_log_probs, actions[:, None], axis=1).squeeze()

    ratio = jnp.exp(action_log_probs - old_action_log_probs)
    clipped_ratio = jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps)

    policy_loss = -jnp.mean(jnp.minimum(ratio * advantages, clipped_ratio * advantages))
    value_loss = jnp.mean((values - returns) ** 2)

    # Compute entropy only over valid actions (following MaskablePPO)
    probs = jax.nn.softmax(masked_logits)
    p_log_p = probs * log_probs
    if avail_mask is not None:
        p_log_p = jnp.where(avail_mask, p_log_p, 0.0)
    entropy = -jnp.mean(jnp.sum(p_log_p, axis=-1))

    return policy_loss + vf_coef * value_loss - ent_coef * entropy


@jax.jit
def update_agent(train_state, obs, actions, old_logits, advantages, returns, avail_mask=None, ent_coef=0.0):
    """Update agent with PPO."""
    loss, grads = jax.value_and_grad(ppo_loss)(
        train_state.params, train_state.apply_fn,
        obs, actions, old_logits, advantages, returns,
        avail_mask=avail_mask, ent_coef=ent_coef
    )
    train_state = train_state.apply_gradients(grads=grads)
    return train_state, loss


def save_policy(params, filepath):
    """Save policy parameters."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump({'params': params}, f)
    print(f"Saved policy to {filepath}")


def extract_episode_stats(rewards, dones):
    """Extract completed episode returns and lengths from a rollout.

    Args:
        rewards: Array of shape (num_steps, num_envs)
        dones: Array of shape (num_steps, num_envs), True at episode boundaries

    Returns:
        Tuple of (episode_returns, episode_lengths) for completed episodes
    """
    # Transfer to CPU once (avoid per-element GPU->CPU transfers)
    rewards_np = np.asarray(rewards)
    dones_np = np.asarray(dones)

    num_steps, num_envs = rewards_np.shape
    episode_returns = []
    episode_lengths = []
    current_returns = np.zeros(num_envs)
    current_lengths = np.zeros(num_envs, dtype=np.int32)

    for t in range(num_steps):
        current_returns += rewards_np[t]
        current_lengths += 1
        done_mask = dones_np[t]
        if np.any(done_mask):
            episode_returns.extend(current_returns[done_mask].tolist())
            episode_lengths.extend(current_lengths[done_mask].tolist())
            current_returns[done_mask] = 0.0
            current_lengths[done_mask] = 0

    return episode_returns, episode_lengths


def get_git_commit():
    """Get current git commit hash, or None if not in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5
        )
        return result.stdout.strip()[:8] if result.returncode == 0 else None
    except Exception:
        return None


def setup_experiment(args):
    """Create experiment directory and save config."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    exp_name = f"{timestamp}_blue-vs-bline_seed{args.seed}"
    exp_dir = Path(args.experiment_dir) / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    git_commit = get_git_commit()

    config = {
        "experiment_name": exp_name,
        "seed": args.seed,
        "timestamp": datetime.now().isoformat(),
        "git_commit": git_commit,
        "hyperparameters": {
            "num_envs": args.num_envs,
            "total_timesteps": args.total_timesteps,
            "rollout_steps": args.rollout_steps,
            "ppo_epochs": args.ppo_epochs,
            "minibatch_size": args.minibatch_size,
            "lr": args.lr,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_eps": 0.2,
            "ent_coef": args.ent_coef,
            "max_grad_norm": args.max_grad_norm,
        },
        "environment": {
            "scenario": "scenario2",
            "max_steps": 100,
            "red_agent": "bline",
        },
        "network": {
            "hidden_dims": [args.hidden_dim, args.hidden_dim],
            "activation": args.activation,
        },
    }

    with open(exp_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    env_info = [
        f"jax_version: {jax.__version__}",
        f"devices: {[str(d) for d in jax.devices()]}",
        f"platform: {jax.default_backend()}",
    ]
    with open(exp_dir / "environment.txt", "w") as f:
        f.write("\n".join(env_info))

    reproduce_script = f"""#!/bin/bash
# Reproduce experiment: {exp_name}
# Generated: {datetime.now().isoformat()}

cd {Path(__file__).parent.parent.resolve()}
{f'git checkout {git_commit}' if git_commit else '# no git commit recorded'}

python scripts/train_cage_vs_bline.py \\
  --seed {args.seed} \\
  --num_envs {args.num_envs} \\
  --total_timesteps {args.total_timesteps} \\
  --rollout_steps {args.rollout_steps} \\
  --ppo_epochs {args.ppo_epochs} \\
  --minibatch_size {args.minibatch_size} \\
  --lr {args.lr} \\
  --ent_coef {args.ent_coef} \\
  --max_grad_norm {args.max_grad_norm} \\
  --hidden_dim {args.hidden_dim} \\
  --activation {args.activation} \\
  --experiment_dir {args.experiment_dir}
"""
    with open(exp_dir / "reproduce.sh", "w") as f:
        f.write(reproduce_script)

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("cage-training")
    mlflow.start_run(run_name=exp_name)
    mlflow.set_tag("codebase", "jaxmarl")
    mlflow.log_params({
        "policy_type": "MlpPolicy",
        "seed": args.seed,
        "num_envs": args.num_envs,
        "total_timesteps": args.total_timesteps,
        "rollout_steps": args.rollout_steps,
        "ppo_epochs": args.ppo_epochs,
        "minibatch_size": args.minibatch_size,
        "learning_rate": args.lr,
        "ent_coef": args.ent_coef,
        "max_grad_norm": args.max_grad_norm,
        "hidden_dim": args.hidden_dim,
        "activation": args.activation,
        "scenario": "Scenario2",
        "red_agent": "bline",
    })

    return exp_dir, config


class MetricsLogger:
    """Logs to both JSONL file and MLflow."""

    def __init__(self, filepath):
        self.filepath = Path(filepath)
        self.file = open(self.filepath, "w")

    def log(self, metrics: dict):
        self.file.write(json.dumps(metrics) + "\n")
        self.file.flush()
        if "final" not in metrics:
            step = metrics.get("steps", metrics.get("update", 0))
            mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))}, step=step)

    def close(self):
        self.file.close()
        mlflow.log_artifact(str(self.filepath))
        mlflow.end_run()


def train(args):
    """Main training loop."""
    exp_dir, config = setup_experiment(args)
    metrics_logger = MetricsLogger(exp_dir / "metrics.jsonl")

    print("=" * 60)
    print("CAGE-JAX PPO Training: Blue vs B_lineAgent")
    print("=" * 60)
    print(f"Experiment dir: {exp_dir}")
    print(f"JAX devices: {jax.devices()}")
    print(f"Num envs: {args.num_envs}, rollout_steps: {args.rollout_steps}")
    print(f"Total timesteps: {args.total_timesteps:,}")
    print(f"Network: [{args.hidden_dim}, {args.hidden_dim}] {args.activation}")
    print(f"PPO: epochs={args.ppo_epochs}, minibatch={args.minibatch_size}, ent_coef={args.ent_coef}")
    print(f"Optimizer: lr={args.lr}, max_grad_norm={args.max_grad_norm}")
    print("=" * 60)

    key = jax.random.PRNGKey(args.seed)
    env = CageEnv(max_steps=100)

    key, key_blue = jax.random.split(key)
    train_state_blue = create_train_state(
        key_blue, BLUE_OBS_DIM, NUM_BLUE_ACTIONS, args.lr,
        hidden_dim=args.hidden_dim, activation=args.activation,
        max_grad_norm=args.max_grad_norm
    )

    # Initialize environments
    key, *env_keys = jax.random.split(key, args.num_envs + 1)
    env_keys = jnp.array(env_keys)
    _, env_states = jax.vmap(env.reset)(env_keys)

    # Initialize B_lineAgent states
    bline_states = bline_reset_batched(args.num_envs)

    total_steps = 0
    episode_returns_blue = []
    episode_lengths_blue = []

    print("\nTraining Blue against B_lineAgent...")
    start_time = time.perf_counter()

    num_updates = args.total_timesteps // (args.num_envs * args.rollout_steps)

    for update in range(num_updates):
        key, env_states, _, bline_states, transitions = collect_rollout(
            key, env, env_states, train_state_blue, bline_states, args.rollout_steps
        )

        total_steps += args.num_envs * args.rollout_steps

        adv_blue, ret_blue = compute_gae(
            transitions['reward_blue'], transitions['value_blue'], transitions['done']
        )

        adv_blue = (adv_blue - adv_blue.mean()) / (adv_blue.std() + 1e-8)

        batch_size = args.rollout_steps * args.num_envs

        obs_blue = transitions['obs_blue'].reshape(batch_size, -1)
        actions_blue = transitions['action_blue'].reshape(batch_size)
        logits_blue = transitions['logits_blue'].reshape(batch_size, -1)
        avail_blue = transitions['avail_blue'].reshape(batch_size, -1)
        adv_blue_flat = adv_blue.reshape(batch_size)
        ret_blue_flat = ret_blue.reshape(batch_size)

        if args.minibatch_size <= 0 or args.minibatch_size >= batch_size:
            minibatch_size = batch_size
            num_minibatches = 1
        else:
            minibatch_size = args.minibatch_size
            num_minibatches = batch_size // minibatch_size

        for _ in range(args.ppo_epochs):
            key, key_perm = jax.random.split(key)
            perm = jax.random.permutation(key_perm, batch_size)

            for mb_idx in range(num_minibatches):
                mb_start = mb_idx * minibatch_size
                mb_end = mb_start + minibatch_size
                mb_indices = perm[mb_start:mb_end]

                train_state_blue, loss_blue = update_agent(
                    train_state_blue,
                    obs_blue[mb_indices],
                    actions_blue[mb_indices],
                    logits_blue[mb_indices],
                    adv_blue_flat[mb_indices],
                    ret_blue_flat[mb_indices],
                    avail_mask=avail_blue[mb_indices],
                    ent_coef=args.ent_coef,
                )

        completed_returns, completed_lengths = extract_episode_stats(
            transitions['reward_blue'], transitions['done']
        )
        episode_returns_blue.extend(completed_returns)
        episode_lengths_blue.extend(completed_lengths)

        if (update + 1) % 10 == 0 or update == 0:
            elapsed = time.perf_counter() - start_time
            sps = total_steps / elapsed

            recent_blue = float(jnp.mean(jnp.array(episode_returns_blue[-100:])))
            recent_len = float(jnp.mean(jnp.array(episode_lengths_blue[-100:]))) if episode_lengths_blue else 0.0

            metrics_logger.log({
                "update": update + 1,
                "steps": total_steps,
                "sps": round(sps),
                "ep_rew_mean": round(recent_blue, 2),
                "ep_len_mean": round(recent_len, 2),
                "loss": round(float(loss_blue), 4),
                "elapsed_sec": round(elapsed, 1),
            })

            print(f"Update {update+1}/{num_updates} | "
                  f"Steps: {total_steps:,} | "
                  f"SPS: {sps:.0f} | "
                  f"Blue Reward: {recent_blue:.1f}")

    elapsed = time.perf_counter() - start_time

    final_metrics = {
        "total_steps": total_steps,
        "wall_time_sec": round(elapsed, 1),
        "throughput_sps": round(total_steps / elapsed),
        "final_blue_reward": round(float(jnp.mean(jnp.array(episode_returns_blue[-100:]))), 2),
    }
    metrics_logger.log({"final": final_metrics})
    metrics_logger.close()

    print("\n" + "=" * 60)
    print(f"Training complete!")
    print(f"Total steps: {total_steps:,}")
    print(f"Wall time: {elapsed:.1f}s")
    print(f"Throughput: {total_steps/elapsed:,.0f} steps/sec")
    print(f"Experiment saved to: {exp_dir}")
    print("=" * 60)

    checkpoint_path = exp_dir / "checkpoint_final.pkl"
    save_policy(train_state_blue.params, checkpoint_path)

    return train_state_blue


def main():
    parser = argparse.ArgumentParser(description="Train Blue agent against B_lineAgent")
    parser.add_argument("--num_envs", type=int, default=1,
                        help="Number of parallel environments (SB3 default: 1)")
    parser.add_argument("--total_timesteps", type=int, default=3_000_000)
    parser.add_argument("--rollout_steps", type=int, default=2048,
                        help="Steps per rollout before PPO update (SB3 default: 2048)")
    parser.add_argument("--ppo_epochs", type=int, default=10,
                        help="PPO epochs per update (SB3 default: 10)")
    parser.add_argument("--minibatch_size", type=int, default=64,
                        help="Minibatch size for PPO updates (SB3 default: 64, 0 = full batch)")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--ent_coef", type=float, default=0.0,
                        help="Entropy coefficient (SB3 default: 0.0)")
    parser.add_argument("--max_grad_norm", type=float, default=0.5,
                        help="Max gradient norm for clipping (SB3 default: 0.5, 0 = no clipping)")
    parser.add_argument("--hidden_dim", type=int, default=256,
                        help="Hidden layer dimension for actor-critic network")
    parser.add_argument("--activation", type=str, default="tanh",
                        choices=["relu", "tanh"],
                        help="Activation function for actor-critic network")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--experiment_dir", type=str, default="experiments",
                        help="Base directory for experiment outputs")
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
