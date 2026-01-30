"""
IPPO (Independent PPO) for CAGE with HeuristicRedCAGE.

Based on the PureJaxRL Implementation of PPO and ippo_rnn_smax.py.
Trains Blue agents against scripted B_lineAgent using feedforward networks.

View experiments:
    mlflow ui  # then open http://localhost:5000
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Dict
from flax.training.train_state import TrainState
import distrax
import hydra
from omegaconf import OmegaConf
import mlflow
import os
import sys
import json
import pickle
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils.cyborg_eval import (
    setup_cyborg_eval,
    evaluate_in_cyborg,
    log_cyborg_eval_results,
)
from jaxmarl.environments.cage.actions import BLUE_ACTION_NAMES

import jaxmarl
from jaxmarl.wrappers.baselines import LogWrapper


class MetricsLogger:
    """Logs to both JSONL file and MLflow."""

    def __init__(self, filepath):
        self.filepath = Path(filepath)
        self.file = open(self.filepath, "w")

    def log(self, metrics: dict, step: int = None):
        self.file.write(json.dumps(metrics) + "\n")
        self.file.flush()
        if step is not None:
            mlflow.log_metrics(
                {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float, np.floating))},
                step=step
            )

    def close(self):
        self.file.close()
        mlflow.log_artifact(str(self.filepath))


class ActorCritic(nn.Module):
    """Feedforward actor-critic network with action masking."""
    action_dim: Sequence[int]
    hidden_dim: int = 64
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x, avail_actions=None):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        actor_mean = nn.Dense(
            self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = activation(actor_mean)
        action_logits = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)

        if avail_actions is not None:
            unavail_actions = 1 - avail_actions
            action_logits = action_logits - (unavail_actions * 1e10)

        pi = distrax.Categorical(logits=action_logits)

        critic = nn.Dense(
            self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        critic = activation(critic)
        critic = nn.Dense(
            self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    avail_actions: jnp.ndarray
    info: jnp.ndarray


def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


def make_train(config):
    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    env = LogWrapper(env)

    def linear_schedule(count):
        frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
        return config["LR"] * frac

    def train(rng):
        network = ActorCritic(
            env.action_space(env.agents[0]).n,
            hidden_dim=config.get("HIDDEN_DIM", 64),
            activation=config["ACTIVATION"]
        )
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros(env.observation_space(env.agents[0]).shape)
        init_avail = jnp.ones(env.action_space(env.agents[0]).n)
        network_params = network.init(_rng, init_x, init_avail)

        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5)
            )

        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset)(reset_rng)

        def _update_step(runner_state, unused):
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, rng = runner_state

                obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])

                # No action masking - policy learns valid actions from experience
                rng, _rng = jax.random.split(rng)
                pi, value = network.apply(train_state.params, obs_batch, None)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                env_act = unbatchify(action, env.agents, config["NUM_ENVS"], env.num_agents)
                env_act = {k: v.squeeze() for k, v in env_act.items()}

                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(env.step)(
                    rng_step, env_state, env_act,
                )

                info = jax.tree.map(lambda x: x.reshape((config["NUM_ACTORS"])), info)
                # Dummy avail_actions (not used for masking, just to keep Transition structure)
                dummy_avail = jnp.ones((config["NUM_ACTORS"], env.action_space(env.agents[0]).n))
                transition = Transition(
                    batchify(done, env.agents, config["NUM_ACTORS"]).squeeze(),
                    action,
                    value,
                    batchify(reward, env.agents, config["NUM_ACTORS"]).squeeze(),
                    log_prob,
                    obs_batch,
                    dummy_avail,
                    info,
                )
                runner_state = (train_state, env_state, obsv, rng)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            train_state, env_state, last_obs, rng = runner_state
            last_obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
            _, last_val = network.apply(train_state.params, last_obs_batch, None)

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=8,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets):
                        pi, value = network.apply(params, traj_batch.obs, None)
                        log_prob = pi.log_prob(traj_batch.action)

                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        logratio = log_prob - traj_batch.log_prob
                        approx_kl = jnp.mean((ratio - 1) - logratio)
                        clip_frac = jnp.mean(jnp.abs(ratio - 1) > config["CLIP_EPS"])
                        var_targets = jnp.var(targets)
                        explained_var = jnp.where(
                            var_targets > 0,
                            1 - jnp.var(targets - value) / var_targets,
                            0.0
                        )

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy, approx_kl, clip_frac, explained_var)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)

                    loss_info = {
                        "total_loss": total_loss[0],
                        "actor_loss": total_loss[1][1],
                        "critic_loss": total_loss[1][0],
                        "entropy": total_loss[1][2],
                        "approx_kl": total_loss[1][3],
                        "clip_frac": total_loss[1][4],
                        "explained_var": total_loss[1][5],
                    }

                    return train_state, loss_info

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree.map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )
                shuffled_batch = jax.tree.map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree.map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )
                train_state, loss_info = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (train_state, traj_batch, advantages, targets, rng)
                return update_state, loss_info

            def callback(metric, update_idx):
                pass  # Metrics collected via scan, logged in main

            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            metric = traj_batch.info
            rng = update_state[-1]

            loss_info = jax.tree.map(lambda x: x.mean(), loss_info)
            metric = jax.tree.map(lambda x: x.mean(), metric)
            metric = {**metric, **loss_info}
            jax.experimental.io_callback(callback, None, metric, update_state[0].step)
            runner_state = (train_state, env_state, last_obs, rng)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, _rng)
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metric}

    return train


@hydra.main(version_base=None, config_path="config", config_name="ippo_ff_cage")
def main(config):
    import time
    config = OmegaConf.to_container(config)

    exp_dir = Path(config.get("EXPERIMENT_DIR", "experiments"))
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    exp_name = f"{timestamp}_ippo_ff_cage_seed{config['SEED']}"
    save_dir = exp_dir / exp_name
    save_dir.mkdir(parents=True, exist_ok=True)

    eval_interval = config.get("EVAL_INTERVAL", 0)
    eval_episodes = config.get("EVAL_EPISODES", 10)
    cyborg_path = config.get("CYBORG_PATH", "/home/paulhax/src/cyber/cage-challenge-2/CybORG")

    cyborg_eval_enabled = False
    if eval_interval > 0:
        if setup_cyborg_eval(cyborg_path):
            cyborg_eval_enabled = True
            print(f"CybORG evaluation enabled (final only, interval={eval_interval})")
        else:
            print("Warning: CybORG evaluation requested but CybORG not available")

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns"))
    mlflow.set_experiment(config.get("MLFLOW_EXPERIMENT", "cage-training"))
    mlflow.start_run(run_name=exp_name)

    mlflow.log_params({
        "algorithm": "IPPO-FF",
        "seed": config["SEED"],
        "num_envs": config["NUM_ENVS"],
        "num_steps": config["NUM_STEPS"],
        "total_timesteps": config["TOTAL_TIMESTEPS"],
        "update_epochs": config["UPDATE_EPOCHS"],
        "num_minibatches": config["NUM_MINIBATCHES"],
        "learning_rate": config["LR"],
        "gamma": config["GAMMA"],
        "gae_lambda": config["GAE_LAMBDA"],
        "clip_eps": config["CLIP_EPS"],
        "ent_coef": config["ENT_COEF"],
        "vf_coef": config["VF_COEF"],
        "max_grad_norm": config["MAX_GRAD_NORM"],
        "hidden_dim": config.get("HIDDEN_DIM", 64),
        "activation": config["ACTIVATION"],
        "anneal_lr": config["ANNEAL_LR"],
        "env_name": config["ENV_NAME"],
        "scenario": config["ENV_KWARGS"].get("scenario", "Scenario2"),
        "max_steps": config["ENV_KWARGS"].get("max_steps", 100),
        "eval_interval": eval_interval,
        "eval_episodes": eval_episodes,
        "cyborg_path": cyborg_path,
    })

    print("=" * 60)
    print("IPPO-FF CAGE Training: Blue vs B_lineAgent")
    print("=" * 60)
    print(f"Environment: {config['ENV_NAME']}")
    print(f"Scenario: {config['ENV_KWARGS'].get('scenario', 'Scenario2')}")
    print(f"Total timesteps: {config['TOTAL_TIMESTEPS']:,}")
    print(f"Num envs: {config['NUM_ENVS']}")
    print(f"Num steps: {config['NUM_STEPS']}")
    print(f"Hidden dim: {config.get('HIDDEN_DIM', 64)}")
    print(f"Activation: {config['ACTIVATION']}")
    print(f"Ent coef: {config['ENT_COEF']}")
    print(f"Seeds: {config.get('NUM_SEEDS', 1)}")
    print(f"Experiment dir: {save_dir}")
    if cyborg_eval_enabled:
        print(f"CybORG eval: {eval_episodes} episodes")
    print("=" * 60)

    start_time = time.perf_counter()

    rng = jax.random.PRNGKey(config["SEED"])
    num_seeds = config.get("NUM_SEEDS", 1)

    if num_seeds > 1:
        rngs = jax.random.split(rng, num_seeds)
        train_jit = jax.jit(make_train(config))
        out = jax.vmap(train_jit)(rngs)
    else:
        train_jit = jax.jit(make_train(config))
        out = train_jit(rng)

    elapsed = time.perf_counter() - start_time
    total_steps = int(config["TOTAL_TIMESTEPS"])
    sps = total_steps / elapsed

    with open(save_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    metrics_logger = MetricsLogger(save_dir / "metrics.jsonl")

    metrics = out["metrics"]
    num_updates = int(config["NUM_UPDATES"])
    steps_per_update = config["NUM_ENVS"] * config["NUM_STEPS"]

    for update_idx in range(num_updates):
        step = (update_idx + 1) * steps_per_update
        update_metrics = {
            "update": update_idx + 1,
            "steps": step,
            "returned_episode_returns": float(metrics["returned_episode_returns"][update_idx].mean()),
            "returned_episode_lengths": float(metrics["returned_episode_lengths"][update_idx].mean()),
            "total_loss": float(metrics["total_loss"][update_idx].mean()),
            "actor_loss": float(metrics["actor_loss"][update_idx].mean()),
            "critic_loss": float(metrics["critic_loss"][update_idx].mean()),
            "entropy": float(metrics["entropy"][update_idx].mean()),
            "approx_kl": float(metrics["approx_kl"][update_idx].mean()),
            "clip_frac": float(metrics["clip_frac"][update_idx].mean()),
            "explained_var": float(metrics["explained_var"][update_idx].mean()),
        }
        metrics_logger.log(update_metrics, step=step)

    metrics_logger.close()

    if num_seeds > 1:
        params = jax.tree.map(lambda x: x[0], out["runner_state"][0][0].params)
    else:
        params = out["runner_state"][0].params

    checkpoint_path = save_dir / "checkpoint_final.pkl"
    with open(checkpoint_path, "wb") as f:
        pickle.dump({"params": params}, f)

    mlflow.log_artifact(str(checkpoint_path), artifact_path="checkpoints")
    mlflow.log_artifact(str(save_dir / "config.json"))

    final_return = float(metrics["returned_episode_returns"][-1].mean())
    final_entropy = float(metrics["entropy"][-1].mean())
    mlflow.log_metrics({
        "final/episode_return": final_return,
        "final/entropy": final_entropy,
        "final/wall_time_sec": elapsed,
        "final/throughput_sps": sps,
    }, step=total_steps)

    print(f"\nTraining complete!")
    print(f"Wall time: {elapsed:.1f}s")
    print(f"Throughput: {sps:,.0f} steps/sec")
    print(f"Final returns: {final_return:.2f}")
    print(f"Final entropy: {final_entropy:.4f}")
    print(f"Saved to: {save_dir}")

    if cyborg_eval_enabled:
        print("\nRunning final CybORG evaluation...")
        eval_start = time.perf_counter()
        cia_results = evaluate_in_cyborg(
            str(checkpoint_path), cyborg_path,
            episodes=eval_episodes * 2,
            steps=100, seed=config["SEED"]
        )
        eval_time = time.perf_counter() - eval_start

        if cia_results:
            print(f"\nFinal CybORG Results ({eval_time:.1f}s):")
            log_cyborg_eval_results(cia_results, mlflow, total_steps)

    mlflow.end_run()
    print(f"\nView in MLflow: mlflow ui")


if __name__ == "__main__":
    main()
