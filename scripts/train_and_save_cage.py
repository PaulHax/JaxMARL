#!/usr/bin/env python
"""Train Blue agent using IPPO on CAGE-JAX and save for CybORG evaluation.

Usage:
    python scripts/train_and_save_cage.py
    python scripts/train_and_save_cage.py --total_timesteps 5000000 --save_path checkpoints/blue_policy.pkl
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
from jaxmarl.environments.cage.actions import NUM_BLUE_ACTIONS, NUM_RED_ACTIONS
from jaxmarl.environments.cage.observations import BLUE_OBS_DIM, RED_OBS_DIM


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
def collect_rollout(key, env, states, train_state_blue, train_state_red, num_steps=128):
    """Collect rollout data from parallel environments."""

    def step_fn(carry, _):
        key, env_states, obs = carry

        key, key_blue, key_red, key_step = jax.random.split(key, 4)

        blue_logits, blue_values = train_state_blue.apply_fn(
            train_state_blue.params, obs['blue']
        )
        red_logits, red_values = train_state_red.apply_fn(
            train_state_red.params, obs['red']
        )

        avail = jax.vmap(env.get_avail_actions)(env_states)

        blue_actions = jax.vmap(sample_action)(
            jax.random.split(key_blue, env_states.time.shape[0]),
            blue_logits,
            avail['blue']
        )
        red_actions = jax.vmap(sample_action)(
            jax.random.split(key_red, env_states.time.shape[0]),
            red_logits,
            avail['red']
        )

        actions = {'blue': blue_actions, 'red': red_actions}

        keys_step = jax.random.split(key_step, env_states.time.shape[0])
        next_obs, next_states, rewards, dones, infos = jax.vmap(env.step)(
            keys_step, env_states, actions
        )

        transition = {
            'obs_blue': obs['blue'],
            'obs_red': obs['red'],
            'action_blue': blue_actions,
            'action_red': red_actions,
            'reward_blue': rewards['blue'],
            'reward_red': rewards['red'],
            'done': dones['__all__'],
            'value_blue': blue_values,
            'value_red': red_values,
            'logits_blue': blue_logits,
            'logits_red': red_logits,
        }

        return (key, next_states, next_obs), transition

    obs = jax.vmap(env.get_obs)(states)

    (key, final_states, final_obs), transitions = jax.lax.scan(
        step_fn, (key, states, obs), None, length=num_steps
    )

    return key, final_states, final_obs, transitions


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
    print("CAGE-JAX IPPO Training (with save)")
    print("=" * 60)
    print(f"JAX devices: {jax.devices()}")
    print(f"Num parallel envs: {args.num_envs}")
    print(f"Total timesteps: {args.total_timesteps:,}")
    print(f"Save path: {args.save_path}")
    print("=" * 60)

    key = jax.random.PRNGKey(args.seed)
    env = CageEnv(max_steps=100)

    key, key_blue, key_red = jax.random.split(key, 3)
    train_state_blue = create_train_state(key_blue, BLUE_OBS_DIM, NUM_BLUE_ACTIONS, args.lr)
    train_state_red = create_train_state(key_red, RED_OBS_DIM, NUM_RED_ACTIONS, args.lr)

    key, *env_keys = jax.random.split(key, args.num_envs + 1)
    env_keys = jnp.array(env_keys)
    _, env_states = jax.vmap(env.reset)(env_keys)

    total_steps = 0
    episode_returns_blue = []
    episode_returns_red = []

    print("\nTraining...")
    start_time = time.perf_counter()

    num_updates = args.total_timesteps // (args.num_envs * args.rollout_steps)

    for update in range(num_updates):
        key, env_states, _, transitions = collect_rollout(
            key, env, env_states, train_state_blue, train_state_red, args.rollout_steps
        )

        total_steps += args.num_envs * args.rollout_steps

        adv_blue, ret_blue = compute_gae(
            transitions['reward_blue'], transitions['value_blue'], transitions['done']
        )
        adv_red, ret_red = compute_gae(
            transitions['reward_red'], transitions['value_red'], transitions['done']
        )

        adv_blue = (adv_blue - adv_blue.mean()) / (adv_blue.std() + 1e-8)
        adv_red = (adv_red - adv_red.mean()) / (adv_red.std() + 1e-8)

        batch_size = args.rollout_steps * args.num_envs

        obs_blue = transitions['obs_blue'].reshape(batch_size, -1)
        obs_red = transitions['obs_red'].reshape(batch_size, -1)
        actions_blue = transitions['action_blue'].reshape(batch_size)
        actions_red = transitions['action_red'].reshape(batch_size)
        logits_blue = transitions['logits_blue'].reshape(batch_size, -1)
        logits_red = transitions['logits_red'].reshape(batch_size, -1)
        adv_blue_flat = adv_blue.reshape(batch_size)
        adv_red_flat = adv_red.reshape(batch_size)
        ret_blue_flat = ret_blue.reshape(batch_size)
        ret_red_flat = ret_red.reshape(batch_size)

        for _ in range(args.ppo_epochs):
            train_state_blue, loss_blue = update_agent(
                train_state_blue, obs_blue, actions_blue, logits_blue, adv_blue_flat, ret_blue_flat
            )
            train_state_red, loss_red = update_agent(
                train_state_red, obs_red, actions_red, logits_red, adv_red_flat, ret_red_flat
            )

        episode_returns_blue.extend(transitions['reward_blue'].sum(axis=0).tolist())
        episode_returns_red.extend(transitions['reward_red'].sum(axis=0).tolist())

        if (update + 1) % 10 == 0 or update == 0:
            elapsed = time.perf_counter() - start_time
            sps = total_steps / elapsed

            recent_blue = jnp.mean(jnp.array(episode_returns_blue[-100:]))
            recent_red = jnp.mean(jnp.array(episode_returns_red[-100:]))

            print(f"Update {update+1}/{num_updates} | "
                  f"Steps: {total_steps:,} | "
                  f"SPS: {sps:.0f} | "
                  f"Blue: {recent_blue:.1f} | "
                  f"Red: {recent_red:.1f}")

    elapsed = time.perf_counter() - start_time
    print("\n" + "=" * 60)
    print(f"Training complete!")
    print(f"Total steps: {total_steps:,}")
    print(f"Wall time: {elapsed:.1f}s")
    print(f"Throughput: {total_steps/elapsed:,.0f} steps/sec")
    print("=" * 60)

    # Save policies
    save_policy(train_state_blue.params, args.save_path)
    if args.save_red:
        red_path = args.save_path.replace('.pkl', '_red.pkl')
        save_policy(train_state_red.params, red_path)

    return train_state_blue, train_state_red


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=1024)
    parser.add_argument("--total_timesteps", type=int, default=5_000_000)
    parser.add_argument("--rollout_steps", type=int, default=128)
    parser.add_argument("--ppo_epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_path", type=str, default="checkpoints/blue_jax_policy.pkl")
    parser.add_argument("--save_red", action="store_true", help="Also save red policy")
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
