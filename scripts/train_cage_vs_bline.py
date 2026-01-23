#!/usr/bin/env python
"""Train Blue agent using PPO against fixed B_lineAgent Red.

This script trains a Blue agent against the deterministic B_lineAgent,
matching the training setup from the cage-2 fork for reproducibility.

Usage:
    python scripts/train_cage_vs_bline.py
    python scripts/train_cage_vs_bline.py --total_timesteps 5000000 --save_path checkpoints/blue_vs_bline.pkl
"""

import argparse
import time
import pickle
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
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

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        logits = nn.Dense(self.action_dim)(x)
        value = nn.Dense(1)(x)
        return logits, value.squeeze(-1)


def create_train_state(key, obs_dim, action_dim, learning_rate=3e-4):
    """Create training state for an agent."""
    network = ActorCritic(action_dim=action_dim)
    dummy_obs = jnp.zeros((1, obs_dim))
    params = network.init(key, dummy_obs)
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


def ppo_loss(params, apply_fn, obs, actions, old_logits, advantages, returns, clip_eps=0.2):
    """PPO clipped objective loss."""
    logits, values = apply_fn(params, obs)

    log_probs = jax.nn.log_softmax(logits)
    old_log_probs = jax.nn.log_softmax(old_logits)

    action_log_probs = jnp.take_along_axis(log_probs, actions[:, None], axis=1).squeeze()
    old_action_log_probs = jnp.take_along_axis(old_log_probs, actions[:, None], axis=1).squeeze()

    ratio = jnp.exp(action_log_probs - old_action_log_probs)
    clipped_ratio = jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps)

    policy_loss = -jnp.mean(jnp.minimum(ratio * advantages, clipped_ratio * advantages))
    value_loss = jnp.mean((values - returns) ** 2)
    entropy = -jnp.mean(jnp.sum(jax.nn.softmax(logits) * log_probs, axis=-1))

    return policy_loss + 0.5 * value_loss - 0.01 * entropy


@jax.jit
def update_agent(train_state, obs, actions, old_logits, advantages, returns):
    """Update agent with PPO."""
    loss, grads = jax.value_and_grad(ppo_loss)(
        train_state.params, train_state.apply_fn,
        obs, actions, old_logits, advantages, returns
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


def train(args):
    """Main training loop."""
    print("=" * 60)
    print("CAGE-JAX PPO Training: Blue vs B_lineAgent")
    print("=" * 60)
    print(f"JAX devices: {jax.devices()}")
    print(f"Num parallel envs: {args.num_envs}")
    print(f"Total timesteps: {args.total_timesteps:,}")
    print(f"Save path: {args.save_path}")
    print("=" * 60)

    key = jax.random.PRNGKey(args.seed)
    env = CageEnv(max_steps=100)

    key, key_blue = jax.random.split(key)
    train_state_blue = create_train_state(key_blue, BLUE_OBS_DIM, NUM_BLUE_ACTIONS, args.lr)

    # Initialize environments
    key, *env_keys = jax.random.split(key, args.num_envs + 1)
    env_keys = jnp.array(env_keys)
    _, env_states = jax.vmap(env.reset)(env_keys)

    # Initialize B_lineAgent states
    bline_states = bline_reset_batched(args.num_envs)

    total_steps = 0
    episode_returns_blue = []

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
        adv_blue_flat = adv_blue.reshape(batch_size)
        ret_blue_flat = ret_blue.reshape(batch_size)

        for _ in range(args.ppo_epochs):
            train_state_blue, loss_blue = update_agent(
                train_state_blue, obs_blue, actions_blue, logits_blue, adv_blue_flat, ret_blue_flat
            )

        episode_returns_blue.extend(transitions['reward_blue'].sum(axis=0).tolist())

        if (update + 1) % 10 == 0 or update == 0:
            elapsed = time.perf_counter() - start_time
            sps = total_steps / elapsed

            recent_blue = jnp.mean(jnp.array(episode_returns_blue[-100:]))

            print(f"Update {update+1}/{num_updates} | "
                  f"Steps: {total_steps:,} | "
                  f"SPS: {sps:.0f} | "
                  f"Blue Reward: {recent_blue:.1f}")

    elapsed = time.perf_counter() - start_time
    print("\n" + "=" * 60)
    print(f"Training complete!")
    print(f"Total steps: {total_steps:,}")
    print(f"Wall time: {elapsed:.1f}s")
    print(f"Throughput: {total_steps/elapsed:,.0f} steps/sec")
    print("=" * 60)

    save_policy(train_state_blue.params, args.save_path)

    return train_state_blue


def main():
    parser = argparse.ArgumentParser(description="Train Blue agent against B_lineAgent")
    parser.add_argument("--num_envs", type=int, default=1024)
    parser.add_argument("--total_timesteps", type=int, default=5_000_000)
    parser.add_argument("--rollout_steps", type=int, default=128)
    parser.add_argument("--ppo_epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_path", type=str, default="checkpoints/blue_vs_bline.pkl")
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
