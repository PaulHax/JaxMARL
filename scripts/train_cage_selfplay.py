#!/usr/bin/env python
"""Train Blue and Red agents using Independent PPO on CAGE-JAX.

This leverages JAX's vmap to run thousands of environments in parallel,
enabling training that would take days with CybORG to complete in minutes.

Usage:
    python scripts/train_cage_ippo.py
    python scripts/train_cage_ippo.py --num_envs 2048 --total_timesteps 10_000_000
"""

import argparse
import time
from functools import partial

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training.train_state import TrainState

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from jaxmarl.environments.cage import CageEnv
from jaxmarl.environments.cage.actions import NUM_BLUE_ACTIONS, NUM_RED_ACTIONS
from jaxmarl.environments.cage.observations import BLUE_OBS_DIM, RED_OBS_DIM


class ActorCritic(nn.Module):
    """Simple actor-critic network."""
    action_dim: int

    @nn.compact
    def __call__(self, x):
        # Shared layers
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)

        # Actor head (policy)
        logits = nn.Dense(self.action_dim)(x)

        # Critic head (value)
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
    # Mask invalid actions
    masked_logits = jnp.where(avail_mask, logits, -1e10)
    return jax.random.categorical(key, masked_logits)


@partial(jax.jit, static_argnums=[1, 5])
def collect_rollout(key, env, states, train_state_blue, train_state_red, num_steps=128):
    """Collect rollout data from parallel environments."""

    def step_fn(carry, _):
        key, env_states, obs = carry

        # Get actions from policies
        key, key_blue, key_red, key_step = jax.random.split(key, 4)

        blue_logits, blue_values = train_state_blue.apply_fn(
            train_state_blue.params, obs['blue']
        )
        red_logits, red_values = train_state_red.apply_fn(
            train_state_red.params, obs['red']
        )

        # Get action masks
        avail = jax.vmap(env.get_avail_actions)(env_states)

        # Sample actions
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

        # Step environments
        keys_step = jax.random.split(key_step, env_states.time.shape[0])
        next_obs, next_states, rewards, dones, infos = jax.vmap(env.step)(
            keys_step, env_states, actions
        )

        # Store transition data
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

    # Get initial observations
    obs = jax.vmap(env.get_obs)(states)

    # Collect rollout
    (key, final_states, final_obs), transitions = jax.lax.scan(
        step_fn, (key, states, obs), None, length=num_steps
    )

    return key, final_states, final_obs, transitions


@jax.jit
def compute_gae(rewards, values, dones, gamma=0.99, gae_lambda=0.95):
    """Compute Generalized Advantage Estimation using JAX scan."""
    # Pad values with zero for bootstrap
    next_values = jnp.concatenate([values[1:], jnp.zeros_like(values[:1])])

    # Compute deltas
    deltas = rewards + gamma * next_values * (1 - dones) - values

    # Reverse scan for GAE
    def scan_fn(lastgae, inputs):
        delta, done = inputs
        advantage = delta + gamma * gae_lambda * (1 - done) * lastgae
        return advantage, advantage

    # Scan backwards
    _, advantages_reversed = jax.lax.scan(
        scan_fn,
        jnp.zeros(rewards.shape[1:]),  # Initial lastgae per env
        (deltas[::-1], dones[::-1])
    )
    advantages = advantages_reversed[::-1]

    returns = advantages + values
    return advantages, returns


def ppo_loss(params, apply_fn, obs, actions, old_logits, advantages, returns, clip_eps=0.2):
    """PPO clipped objective loss."""
    logits, values = apply_fn(params, obs)

    # Policy loss
    log_probs = jax.nn.log_softmax(logits)
    old_log_probs = jax.nn.log_softmax(old_logits)

    action_log_probs = jnp.take_along_axis(log_probs, actions[:, None], axis=1).squeeze()
    old_action_log_probs = jnp.take_along_axis(old_log_probs, actions[:, None], axis=1).squeeze()

    ratio = jnp.exp(action_log_probs - old_action_log_probs)
    clipped_ratio = jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps)

    policy_loss = -jnp.mean(jnp.minimum(ratio * advantages, clipped_ratio * advantages))

    # Value loss
    value_loss = jnp.mean((values - returns) ** 2)

    # Entropy bonus
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


def train(args):
    """Main training loop."""
    print("=" * 60)
    print("CAGE-JAX IPPO Training")
    print("=" * 60)
    print(f"JAX devices: {jax.devices()}")
    print(f"Num parallel envs: {args.num_envs}")
    print(f"Total timesteps: {args.total_timesteps:,}")
    print("=" * 60)

    # Initialize
    key = jax.random.PRNGKey(args.seed)
    env = CageEnv(max_steps=100)

    # Create agent networks
    key, key_blue, key_red = jax.random.split(key, 3)
    train_state_blue = create_train_state(key_blue, BLUE_OBS_DIM, NUM_BLUE_ACTIONS, args.lr)
    train_state_red = create_train_state(key_red, RED_OBS_DIM, NUM_RED_ACTIONS, args.lr)

    # Initialize parallel environments
    key, *env_keys = jax.random.split(key, args.num_envs + 1)
    env_keys = jnp.array(env_keys)
    _, env_states = jax.vmap(env.reset)(env_keys)

    # Training metrics
    total_steps = 0
    episode_returns_blue = []
    episode_returns_red = []

    print("\nTraining...")
    start_time = time.perf_counter()

    num_updates = args.total_timesteps // (args.num_envs * args.rollout_steps)

    for update in range(num_updates):
        # Collect rollout
        key, env_states, _, transitions = collect_rollout(
            key, env, env_states, train_state_blue, train_state_red, args.rollout_steps
        )

        total_steps += args.num_envs * args.rollout_steps

        # Compute advantages for both agents
        adv_blue, ret_blue = compute_gae(
            transitions['reward_blue'], transitions['value_blue'], transitions['done']
        )
        adv_red, ret_red = compute_gae(
            transitions['reward_red'], transitions['value_red'], transitions['done']
        )

        # Normalize advantages
        adv_blue = (adv_blue - adv_blue.mean()) / (adv_blue.std() + 1e-8)
        adv_red = (adv_red - adv_red.mean()) / (adv_red.std() + 1e-8)

        # Flatten batch dimensions
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

        # Update agents
        for _ in range(args.ppo_epochs):
            train_state_blue, loss_blue = update_agent(
                train_state_blue, obs_blue, actions_blue, logits_blue, adv_blue_flat, ret_blue_flat
            )
            train_state_red, loss_red = update_agent(
                train_state_red, obs_red, actions_red, logits_red, adv_red_flat, ret_red_flat
            )

        # Track episode returns
        episode_returns_blue.extend(transitions['reward_blue'].sum(axis=0).tolist())
        episode_returns_red.extend(transitions['reward_red'].sum(axis=0).tolist())

        # Log progress
        if (update + 1) % 5 == 0 or update == 0:
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

    return train_state_blue, train_state_red


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=1024)
    parser.add_argument("--total_timesteps", type=int, default=1_000_000)
    parser.add_argument("--rollout_steps", type=int, default=128)
    parser.add_argument("--ppo_epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
