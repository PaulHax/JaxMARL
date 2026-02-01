#!/usr/bin/env python
"""Train Blue agent using PPO against RedMeanderAgent.

This script trains a Blue agent against the exploratory RedMeanderAgent,
which uses opportunistic exploration rather than a fixed attack path.

Each run creates an experiment directory with:
  - config.json: All hyperparameters and settings
  - metrics.jsonl: Per-update training metrics
  - checkpoint_final.pkl: Trained model weights
  - reproduce.sh: Script to reproduce the experiment
  - environment.txt: JAX version and device info

Optional: Periodic CybORG evaluation with CIA metrics (requires CybORG installed).
  Use --eval_interval to enable and --cyborg_path to specify CybORG location.

Usage:
    python scripts/train_cage_vs_meander.py
    python scripts/train_cage_vs_meander.py --seed 42 --total_timesteps 3000000
    python scripts/train_cage_vs_meander.py --eval_interval 100000 --cyborg_path /path/to/CybORG

View experiments:
    mlflow ui  # then open http://localhost:5000
"""

import argparse
import json
import os
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
import distrax
from flax import linen as nn
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from jaxmarl.environments.cage import CageEnv
from utils.metrics import MetricsLogger
from utils.cyborg_eval import (
    setup_cyborg_eval,
    evaluate_in_cyborg,
    log_cyborg_eval_results,
    run_final_cyborg_eval,
)
from jaxmarl.environments.cage.actions import NUM_BLUE_ACTIONS, compute_blue_action_space_size
from jaxmarl.environments.cage.observations import BLUE_OBS_DIM, compute_blue_obs_dim
from jaxmarl.environments.cage.scripted_agents import (
    MeanderState,
    meander_reset_batched,
    meander_get_action_batched,
)


class ActorCritic(nn.Module):
    """Actor-critic network with orthogonal initialization and action masking (following JaxMARL IPPO)."""
    action_dim: int
    hidden_dim: int = 64
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x, avail_actions=None):
        act_fn = nn.tanh if self.activation == "tanh" else nn.relu
        # Actor
        actor = nn.Dense(self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        actor = act_fn(actor)
        actor = nn.Dense(self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(actor)
        actor = act_fn(actor)
        logits = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor)

        # Apply action masking (following SMAX IPPO pattern)
        if avail_actions is not None:
            unavail_actions = 1 - avail_actions
            logits = logits - (unavail_actions * 1e10)

        pi = distrax.Categorical(logits=logits)

        # Critic
        critic = nn.Dense(self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        critic = act_fn(critic)
        critic = nn.Dense(self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(critic)
        critic = act_fn(critic)
        value = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)
        return pi, value.squeeze(-1)


def create_train_state(key, obs_dim, action_dim, learning_rate=3e-4, hidden_dim=256,
                       activation="tanh", max_grad_norm=0.5, num_updates=1, anneal_lr=True):
    """Create training state for an agent."""
    network = ActorCritic(action_dim=action_dim, hidden_dim=hidden_dim, activation=activation)
    dummy_obs = jnp.zeros((1, obs_dim))
    dummy_avail = jnp.ones((1, action_dim))
    params = network.init(key, dummy_obs, dummy_avail)

    if anneal_lr:
        lr_schedule = optax.linear_schedule(
            init_value=learning_rate,
            end_value=0.0,
            transition_steps=num_updates
        )
    else:
        lr_schedule = learning_rate

    if max_grad_norm > 0:
        tx = optax.chain(
            optax.clip_by_global_norm(max_grad_norm),
            optax.adam(lr_schedule, eps=1e-5),
        )
    else:
        tx = optax.adam(lr_schedule, eps=1e-5)
    return TrainState.create(apply_fn=network.apply, params=params, tx=tx)


@partial(jax.jit, static_argnums=[1, 5, 6])
def collect_rollout(key, env, states, train_state_blue, meander_states, num_steps=128, reward_scale=0.01):
    """Collect rollout data from parallel environments.

    Uses learned Blue policy (with distrax) and fixed RedMeanderAgent for Red.
    Rewards are scaled by reward_scale to improve learning stability.
    """

    def step_fn(carry, _):
        key, env_states, obs, meander_states = carry

        key, key_blue, key_red, key_step, key_meander_reset = jax.random.split(key, 5)

        avail = jax.vmap(env.get_avail_actions)(env_states)

        # Blue: learned policy with action masking via distrax
        pi, blue_values = train_state_blue.apply_fn(
            train_state_blue.params, obs['blue'], None  # No action masking
        )
        blue_actions = pi.sample(seed=key_blue)
        blue_log_probs = pi.log_prob(blue_actions)

        # Red: fixed RedMeanderAgent
        red_keys = jax.random.split(key_red, env_states.time.shape[0])
        red_actions, new_meander_states = meander_get_action_batched(
            meander_states, obs['red'], avail['red'], env.const, red_keys
        )

        actions = {'blue': blue_actions, 'red': red_actions}

        keys_step = jax.random.split(key_step, env_states.time.shape[0])
        next_obs, next_states, rewards, dones, infos = jax.vmap(env.step)(
            keys_step, env_states, actions
        )

        # Reset MeanderAgent state when episode ends
        batch_size = env_states.time.shape[0]
        reset_meander = meander_reset_batched(batch_size, env.const, key_meander_reset)
        new_meander_states = MeanderState(
            scanned_subnets=jnp.where(dones['__all__'][:, None], reset_meander.scanned_subnets, new_meander_states.scanned_subnets),
            scanned_ips=jnp.where(dones['__all__'][:, None], reset_meander.scanned_ips, new_meander_states.scanned_ips),
            exploited_ips=jnp.where(dones['__all__'][:, None], reset_meander.exploited_ips, new_meander_states.exploited_ips),
            escalated_hosts=jnp.where(dones['__all__'][:, None], reset_meander.escalated_hosts, new_meander_states.escalated_hosts),
            host_ip_known=jnp.where(dones['__all__'][:, None], reset_meander.host_ip_known, new_meander_states.host_ip_known),
            last_host_idx=jnp.where(dones['__all__'], reset_meander.last_host_idx, new_meander_states.last_host_idx),
            last_ip_idx=jnp.where(dones['__all__'], reset_meander.last_ip_idx, new_meander_states.last_ip_idx),
            last_action_success=jnp.where(dones['__all__'], reset_meander.last_action_success, new_meander_states.last_action_success),
        )

        transition = {
            'obs_blue': obs['blue'],
            'action_blue': blue_actions,
            'reward_blue': rewards['blue'] * reward_scale,
            'done': dones['__all__'],
            'value_blue': blue_values,
            'log_prob_blue': blue_log_probs,
            'avail_blue': avail['blue'],
        }

        return (key, next_states, next_obs, new_meander_states), transition

    obs = jax.vmap(env.get_obs)(states)

    (key, final_states, final_obs, final_meander_states), transitions = jax.lax.scan(
        step_fn, (key, states, obs, meander_states), None, length=num_steps
    )

    return key, final_states, final_obs, final_meander_states, transitions


@jax.jit
def compute_gae(rewards, values, dones, last_val, gamma=0.99, gae_lambda=0.95):
    """Compute Generalized Advantage Estimation with proper bootstrapping.

    Args:
        rewards: (num_steps, num_envs) rewards
        values: (num_steps, num_envs) value estimates
        dones: (num_steps, num_envs) episode done flags
        last_val: (num_envs,) value estimate for final observation (bootstrap)
        gamma: discount factor
        gae_lambda: GAE lambda parameter
    """
    next_values = jnp.concatenate([values[1:], last_val[None, :]])
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


def ppo_loss_with_metrics(params, apply_fn, obs, actions, old_log_probs, old_values, advantages, returns, avail_mask, clip_eps=0.2, vf_coef=0.5, ent_coef=0.01):
    """PPO clipped objective loss with distrax distribution (following JaxMARL IPPO).

    Returns (total_loss, metrics_dict) for logging.
    """
    pi, values = apply_fn(params, obs, avail_mask)
    log_probs = pi.log_prob(actions)

    logratio = log_probs - old_log_probs
    ratio = jnp.exp(logratio)
    clipped_ratio = jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps)
    policy_loss = -jnp.mean(jnp.minimum(ratio * advantages, clipped_ratio * advantages))

    value_pred_clipped = old_values + jnp.clip(values - old_values, -clip_eps, clip_eps)
    value_losses = jnp.square(values - returns)
    value_losses_clipped = jnp.square(value_pred_clipped - returns)
    value_loss = 0.5 * jnp.mean(jnp.maximum(value_losses, value_losses_clipped))

    entropy = pi.entropy().mean()

    # Approx KL divergence: mean((ratio - 1) - logratio)
    approx_kl = jnp.mean((ratio - 1) - logratio)

    # Explained variance: how well value function predicts returns
    var_returns = jnp.var(returns)
    explained_var = jnp.where(
        var_returns > 0,
        1 - jnp.var(returns - values) / var_returns,
        0.0
    )

    # Clip fraction: how often ratio is clipped
    clip_frac = jnp.mean(jnp.abs(ratio - 1) > clip_eps)

    total_loss = policy_loss + vf_coef * value_loss - ent_coef * entropy

    metrics = {
        "entropy": entropy,
        "approx_kl": approx_kl,
        "explained_var": explained_var,
        "clip_frac": clip_frac,
        "policy_loss": policy_loss,
        "value_loss": value_loss,
    }

    return total_loss, metrics


def ppo_loss(params, apply_fn, obs, actions, old_log_probs, old_values, advantages, returns, avail_mask, clip_eps=0.2, vf_coef=0.5, ent_coef=0.01):
    """PPO loss (returns only total loss for backward compat)."""
    loss, _ = ppo_loss_with_metrics(
        params, apply_fn, obs, actions, old_log_probs, old_values,
        advantages, returns, avail_mask, clip_eps, vf_coef, ent_coef
    )
    return loss


@jax.jit
def update_agent(train_state, obs, actions, old_log_probs, old_values, advantages, returns, avail_mask, ent_coef=0.01):
    """Update agent with PPO using distrax."""
    advantages = (advantages - jnp.mean(advantages)) / (jnp.std(advantages) + 1e-8)
    loss, grads = jax.value_and_grad(ppo_loss)(
        train_state.params, train_state.apply_fn,
        obs, actions, old_log_probs, old_values, advantages, returns,
        avail_mask, ent_coef=ent_coef
    )
    train_state = train_state.apply_gradients(grads=grads)
    return train_state, loss


@jax.jit
def update_agent_with_metrics(train_state, obs, actions, old_log_probs, old_values, advantages, returns, avail_mask, ent_coef=0.01):
    """Update agent and return detailed metrics for logging."""
    advantages = (advantages - jnp.mean(advantages)) / (jnp.std(advantages) + 1e-8)
    loss, grads = jax.value_and_grad(ppo_loss)(
        train_state.params, train_state.apply_fn,
        obs, actions, old_log_probs, old_values, advantages, returns,
        avail_mask, ent_coef=ent_coef
    )
    train_state = train_state.apply_gradients(grads=grads)

    _, metrics = ppo_loss_with_metrics(
        train_state.params, train_state.apply_fn,
        obs, actions, old_log_probs, old_values, advantages, returns,
        avail_mask, ent_coef=ent_coef
    )
    return train_state, loss, metrics


def save_policy(params, filepath):
    """Save policy parameters."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump({'params': params}, f)
    print(f"Saved policy to {filepath}")


class EpisodeTracker:
    """Track episode returns and lengths across rollout boundaries."""

    def __init__(self, num_envs: int):
        self.num_envs = num_envs
        self.current_returns = np.zeros(num_envs)
        self.current_lengths = np.zeros(num_envs, dtype=np.int32)

    def update(self, rewards, dones):
        """Process a rollout and return completed episode stats.

        Args:
            rewards: Array of shape (num_steps, num_envs)
            dones: Array of shape (num_steps, num_envs), True at episode boundaries

        Returns:
            Tuple of (episode_returns, episode_lengths) for completed episodes
        """
        rewards_np = np.asarray(rewards)
        dones_np = np.asarray(dones)

        num_steps = rewards_np.shape[0]
        episode_returns = []
        episode_lengths = []

        for t in range(num_steps):
            self.current_returns += rewards_np[t]
            self.current_lengths += 1
            done_mask = dones_np[t]
            if np.any(done_mask):
                episode_returns.extend(self.current_returns[done_mask].tolist())
                episode_lengths.extend(self.current_lengths[done_mask].tolist())
                self.current_returns[done_mask] = 0.0
                self.current_lengths[done_mask] = 0

        return episode_returns, episode_lengths


def get_git_info(path: Path):
    """Get git commit and repo root for a path."""
    try:
        repo_root = subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "--show-toplevel"], text=True, timeout=5
        ).strip()
        commit = subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True, timeout=5
        ).strip()
        return repo_root, commit
    except Exception:
        return None, "unknown"


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


def log_reproducibility(script_path: Path):
    """Log reproducibility info as MLflow tags."""
    script_repo, script_commit = get_git_info(script_path.parent)

    mlflow.set_tag("command", " ".join(sys.argv))
    mlflow.set_tag("git_commit", script_commit)
    if script_repo:
        mlflow.set_tag("git_repo", script_repo)


def setup_experiment(args):
    """Create experiment directory and save config."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    exp_name = f"{timestamp}_blue-vs-meander"
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
            "num_minibatches": args.num_minibatches,
            "lr": args.lr,
            "anneal_lr": args.anneal_lr,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_eps": 0.2,
            "ent_coef": args.ent_coef,
            "max_grad_norm": args.max_grad_norm,
        },
        "environment": {
            "scenario": args.scenario,
            "max_steps": args.max_steps,
            "red_agent": "meander",
        },
        "network": {
            "hidden_dims": [args.hidden_dim, args.hidden_dim],
            "activation": args.activation,
        },
        "evaluation": {
            "eval_interval": args.eval_interval,
            "eval_episodes": args.eval_episodes,
            "cyborg_path": args.cyborg_path,
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

    eval_args = ""
    if args.eval_interval > 0:
        eval_args = f"""  --eval_interval {args.eval_interval} \\
  --eval_episodes {args.eval_episodes} \\
  --cyborg_path {args.cyborg_path} \\
"""

    reproduce_script = f"""#!/bin/bash
# Reproduce experiment: {exp_name}
# Generated: {datetime.now().isoformat()}

cd {Path(__file__).parent.parent.resolve()}
{f'git checkout {git_commit}' if git_commit else '# no git commit recorded'}

python scripts/train_cage_vs_meander.py \\
  --seed {args.seed} \\
  --scenario {args.scenario} \\
  --max_steps {args.max_steps} \\
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
{eval_args}  --experiment_dir {args.experiment_dir}
"""
    with open(exp_dir / "reproduce.sh", "w") as f:
        f.write(reproduce_script)

    if not os.environ.get("MLFLOW_TRACKING_URI"):
        cyber_root = Path(__file__).resolve().parent.parent.parent
        mlflow_dir = cyber_root / "mlflow"
        mlflow_dir.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(f"sqlite:///{mlflow_dir / 'mlflow.db'}")
    mlflow.set_experiment("cage-training")
    mlflow.start_run(run_name="blue-vs-meander")
    mlflow.set_tag("codebase", "jaxmarl")
    log_reproducibility(Path(__file__))
    mlflow.log_params({
        "policy_type": "MlpPolicy",
        "seed": args.seed,
        "num_envs": args.num_envs,
        "total_timesteps": args.total_timesteps,
        "rollout_steps": args.rollout_steps,
        "ppo_epochs": args.ppo_epochs,
        "minibatch_size": args.minibatch_size,
        "num_minibatches": args.num_minibatches,
        "learning_rate": args.lr,
        "anneal_lr": args.anneal_lr,
        "ent_coef": args.ent_coef,
        "max_grad_norm": args.max_grad_norm,
        "hidden_dim": args.hidden_dim,
        "activation": args.activation,
        "scenario": args.scenario,
        "red_agent": "meander",
        "eval_interval": args.eval_interval,
        "eval_episodes": args.eval_episodes,
        "cyborg_path": args.cyborg_path,
    })

    return exp_dir, config


def train(args):
    """Main training loop."""
    exp_dir, config = setup_experiment(args)
    metrics_logger = MetricsLogger(exp_dir / "metrics.jsonl")

    cyborg_eval_enabled = False
    if args.eval_interval > 0:
        if setup_cyborg_eval(args.cyborg_path):
            cyborg_eval_enabled = True
            print(f"CybORG evaluation enabled every {args.eval_interval:,} steps")
        else:
            print("Warning: CybORG evaluation requested but CybORG not available")

    key = jax.random.PRNGKey(args.seed)
    env = CageEnv(scenario=args.scenario, max_steps=args.max_steps)

    obs_dim = compute_blue_obs_dim(env.const)
    action_dim = compute_blue_action_space_size(env.const)

    print("=" * 60)
    print("CAGE-JAX PPO Training: Blue vs RedMeanderAgent")
    print("=" * 60)
    print(f"Experiment dir: {exp_dir}")
    print(f"Scenario: {args.scenario} ({env.const.num_hosts} hosts)")
    print(f"Obs dim: {obs_dim}, Action dim: {action_dim}")
    print(f"JAX devices: {jax.devices()}")
    print(f"Num envs: {args.num_envs}, rollout_steps: {args.rollout_steps}")
    print(f"Total timesteps: {args.total_timesteps:,}")
    print(f"Network: [{args.hidden_dim}, {args.hidden_dim}] {args.activation}")
    print(f"PPO: epochs={args.ppo_epochs}, minibatches={args.num_minibatches}, ent_coef={args.ent_coef}")
    print(f"Optimizer: lr={args.lr}, max_grad_norm={args.max_grad_norm}, anneal_lr={args.anneal_lr}")
    if cyborg_eval_enabled:
        print(f"CybORG eval: every {args.eval_interval:,} steps, {args.eval_episodes} episodes")
    print("=" * 60)

    num_updates = args.total_timesteps // (args.num_envs * args.rollout_steps)

    key, key_blue = jax.random.split(key)
    train_state_blue = create_train_state(
        key_blue, obs_dim, action_dim, args.lr,
        hidden_dim=args.hidden_dim, activation=args.activation,
        max_grad_norm=args.max_grad_norm,
        num_updates=num_updates, anneal_lr=args.anneal_lr
    )

    # Initialize environments
    key, key_meander, *env_keys = jax.random.split(key, args.num_envs + 2)
    env_keys = jnp.array(env_keys)
    _, env_states = jax.vmap(env.reset)(env_keys)

    # Initialize RedMeanderAgent states
    meander_states = meander_reset_batched(args.num_envs, env.const, key_meander)

    total_steps = 0
    last_eval_step = 0
    episode_returns_blue = []
    episode_tracker = EpisodeTracker(args.num_envs)

    print("\nTraining Blue against RedMeanderAgent...")
    start_time = time.perf_counter()

    num_updates = args.total_timesteps // (args.num_envs * args.rollout_steps)

    for update in range(num_updates):
        key, env_states, final_obs, meander_states, transitions = collect_rollout(
            key, env, env_states, train_state_blue, meander_states, args.rollout_steps, args.reward_scale
        )

        total_steps += args.num_envs * args.rollout_steps

        # Compute bootstrap value for GAE (value of final observation)
        _, last_val = train_state_blue.apply_fn(
            train_state_blue.params, final_obs['blue'], None
        )

        adv_blue, ret_blue = compute_gae(
            transitions['reward_blue'], transitions['value_blue'], transitions['done'], last_val
        )

        batch_size = args.rollout_steps * args.num_envs

        obs_blue = transitions['obs_blue'].reshape(batch_size, -1)
        actions_blue = transitions['action_blue'].reshape(batch_size)
        log_probs_blue = transitions['log_prob_blue'].reshape(batch_size)
        values_blue = transitions['value_blue'].reshape(batch_size)
        avail_blue = transitions['avail_blue'].reshape(batch_size, -1)
        adv_blue_flat = adv_blue.reshape(batch_size)
        ret_blue_flat = ret_blue.reshape(batch_size)

        if args.minibatch_size > 0:
            minibatch_size = args.minibatch_size
            num_minibatches = batch_size // minibatch_size
        else:
            num_minibatches = args.num_minibatches
            minibatch_size = batch_size // num_minibatches

        ppo_metrics = None
        for epoch in range(args.ppo_epochs):
            key, key_perm = jax.random.split(key)
            perm = jax.random.permutation(key_perm, batch_size)

            for mb_idx in range(num_minibatches):
                mb_start = mb_idx * minibatch_size
                mb_end = mb_start + minibatch_size
                mb_indices = perm[mb_start:mb_end]

                is_last = (epoch == args.ppo_epochs - 1) and (mb_idx == num_minibatches - 1)
                if is_last:
                    train_state_blue, loss_blue, ppo_metrics = update_agent_with_metrics(
                        train_state_blue,
                        obs_blue[mb_indices],
                        actions_blue[mb_indices],
                        log_probs_blue[mb_indices],
                        values_blue[mb_indices],
                        adv_blue_flat[mb_indices],
                        ret_blue_flat[mb_indices],
                        None,  # No action masking for better transfer
                        ent_coef=args.ent_coef,
                    )
                else:
                    train_state_blue, loss_blue = update_agent(
                        train_state_blue,
                        obs_blue[mb_indices],
                        actions_blue[mb_indices],
                        log_probs_blue[mb_indices],
                        values_blue[mb_indices],
                        adv_blue_flat[mb_indices],
                        ret_blue_flat[mb_indices],
                        None,  # No action masking for better transfer
                        ent_coef=args.ent_coef,
                    )

        completed_returns, completed_lengths = episode_tracker.update(
            transitions['reward_blue'], transitions['done']
        )
        episode_returns_blue.extend(completed_returns)

        if (update + 1) % 10 == 0 or update == 0:
            elapsed = time.perf_counter() - start_time
            sps = total_steps / elapsed

            recent_blue = float(jnp.mean(jnp.array(episode_returns_blue[-100:])))
            log_dict = {
                "update": update + 1,
                "steps": total_steps,
                "steps_per_second": round(sps),
                "episode_reward_mean": round(recent_blue, 2),
                "loss": round(float(loss_blue), 4),
            }

            if ppo_metrics is not None:
                log_dict.update({
                    "entropy": round(float(ppo_metrics["entropy"]), 4),
                    "kl_divergence": round(float(ppo_metrics["approx_kl"]), 6),
                    "explained_variance": round(float(ppo_metrics["explained_var"]), 4),
                    "clip_fraction": round(float(ppo_metrics["clip_frac"]), 4),
                    "policy_loss": round(float(ppo_metrics["policy_loss"]), 4),
                    "value_loss": round(float(ppo_metrics["value_loss"]), 4),
                })

            metrics_logger.log(log_dict)

            entropy_str = f" | Entropy: {ppo_metrics['entropy']:.3f}" if ppo_metrics else ""
            print(f"Update {update+1}/{num_updates} | "
                  f"Steps: {total_steps:,} | "
                  f"steps/sec: {sps:.0f} | "
                  f"Blue Reward: {recent_blue:.1f}{entropy_str}")

        if cyborg_eval_enabled and args.eval_interval > 0:
            if total_steps >= last_eval_step + args.eval_interval:
                last_eval_step = total_steps
                checkpoint_path = exp_dir / f"checkpoint_{total_steps}.pkl"
                save_policy(train_state_blue.params, checkpoint_path)

                print(f"\n  Running CybORG evaluation at {total_steps:,} steps...")
                eval_start = time.perf_counter()
                cia_results = evaluate_in_cyborg(
                    str(checkpoint_path), args.cyborg_path,
                    episodes=args.eval_episodes, steps=100, seed=args.seed
                )
                eval_time = time.perf_counter() - eval_start

                if cia_results:
                    mlflow.log_artifact(str(checkpoint_path), artifact_path="checkpoints")
                    mlflow.log_metrics({
                        "eval/confidentiality": cia_results["confidentiality"],
                        "eval/integrity": cia_results["integrity"],
                        "eval/availability": cia_results["availability"],
                        "eval/resilience": cia_results["resilience"],
                        "eval/cyborg_reward": cia_results["reward"],
                    }, step=total_steps)

                    metrics_logger.log({
                        "eval_step": total_steps,
                        "eval/confidentiality": round(cia_results["confidentiality"], 3),
                        "eval/integrity": round(cia_results["integrity"], 3),
                        "eval/availability": round(cia_results["availability"], 3),
                        "eval/resilience": round(cia_results["resilience"], 3),
                        "eval/cyborg_reward": round(cia_results["reward"], 2),
                    })

                    print(f"  CybORG eval ({eval_time:.1f}s): "
                          f"C={cia_results['confidentiality']:.2f} "
                          f"I={cia_results['integrity']:.2f} "
                          f"A={cia_results['availability']:.2f} "
                          f"R={cia_results['resilience']:.2f} "
                          f"Reward={cia_results['reward']:.1f}\n")

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
    mlflow.log_artifact(str(checkpoint_path), artifact_path="checkpoints")

    run_final_cyborg_eval(
        str(checkpoint_path), args.cyborg_path, mlflow, total_steps,
        export_dir=str(exp_dir), episodes=args.eval_episodes, steps=100, seed=args.seed
    )

    return train_state_blue


def main():
    parser = argparse.ArgumentParser(description="Train Blue agent against RedMeanderAgent")
    parser.add_argument("--num_envs", type=int, default=16,
                        help="Number of parallel environments (IPPO default: 16)")
    parser.add_argument("--total_timesteps", type=int, default=3_000_000)
    parser.add_argument("--rollout_steps", type=int, default=128,
                        help="Steps per rollout before PPO update (IPPO default: 128)")
    parser.add_argument("--ppo_epochs", type=int, default=4,
                        help="PPO epochs per update (IPPO: 4)")
    parser.add_argument("--minibatch_size", type=int, default=0,
                        help="Minibatch size for PPO updates (0 = use num_minibatches)")
    parser.add_argument("--num_minibatches", type=int, default=4,
                        help="Number of minibatches for PPO updates (IPPO: 4)")
    parser.add_argument("--anneal_lr", action="store_true", default=True,
                        help="Use linear LR annealing (IPPO default)")
    parser.add_argument("--no_anneal_lr", dest="anneal_lr", action="store_false")
    parser.add_argument("--lr", type=float, default=2.5e-4,
                        help="Learning rate (IPPO: 2.5e-4)")
    parser.add_argument("--ent_coef", type=float, default=0.01,
                        help="Entropy coefficient (IPPO: 0.01)")
    parser.add_argument("--reward_scale", type=float, default=1.0,
                        help="Scale factor for rewards (1.0 = no scaling)")
    parser.add_argument("--max_grad_norm", type=float, default=0.5,
                        help="Max gradient norm for clipping (SB3 default: 0.5, 0 = no clipping)")
    parser.add_argument("--hidden_dim", type=int, default=64,
                        help="Hidden layer dimension for actor-critic network (IPPO: 64)")
    parser.add_argument("--activation", type=str, default="tanh",
                        choices=["relu", "tanh"],
                        help="Activation function for actor-critic network")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scenario", type=str, default="Scenario2",
                        choices=["Scenario2", "hosts_2", "hosts_3", "hosts_4", "hosts_5"],
                        help="Scenario to use (default: Scenario2 with 13 hosts)")
    parser.add_argument("--max_steps", type=int, default=100,
                        help="Max steps per episode (default: 100)")
    parser.add_argument("--experiment_dir", type=str, default="experiments",
                        help="Base directory for experiment outputs")
    parser.add_argument("--eval_interval", type=int, default=0,
                        help="Evaluate in CybORG every N steps (0 = disabled)")
    parser.add_argument("--eval_episodes", type=int, default=10,
                        help="Number of episodes per CybORG evaluation")
    parser.add_argument("--cyborg_path", type=str,
                        default="/home/paulhax/src/cyber/cage-challenge-2/CybORG",
                        help="Path to CybORG installation for CIA evaluation")
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
