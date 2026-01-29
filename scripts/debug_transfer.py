"""Debug script to compare JaxMARL vs CybORG observations and action effects."""

import sys
sys.path.insert(0, '/home/paulhax/src/cyber/cage-challenge-2/CybORG')

import numpy as np
import jax
import jax.numpy as jnp
import pickle

# JaxMARL imports
from jaxmarl.environments.cage.heuristic_red_cage_env import HeuristicRedCAGE
from jaxmarl.environments.cage.observations import get_blue_obs
from jaxmarl.environments.cage.actions import decode_blue_action, NUM_BLUE_ACTIONS, get_blue_action_mask, BLUE_ACTION_NAMES
from jaxmarl.environments.cage.state import CageConst
from jaxmarl.environments.cage.scenarios.scenario2 import create_scenario2_const

# CybORG imports
from CybORG import CybORG
from CybORG.Agents import B_lineAgent, SleepAgent
from CybORG.Agents.Wrappers import BlueTableWrapper
from CybORG.Agents.SimpleAgents.JaxPolicyAgent import JaxPolicyAgent


def get_action_type_name(action_idx):
    """Convert action index to action type name."""
    if action_idx == 0:
        return "Sleep"
    elif action_idx == 1:
        return "Monitor"
    elif action_idx < 15:  # 2-14
        return f"Analyse({action_idx - 2})"
    elif action_idx < 28:  # 15-27
        return f"Remove({action_idx - 15})"
    elif action_idx < 132:  # 28-131
        decoy_idx = action_idx - 28
        decoy_type = decoy_idx // 13
        host = decoy_idx % 13
        decoy_names = ["Apache", "Femitter", "HarakaSMPT", "Smss", "SSHD", "Svchost", "Tomcat", "Vsftpd"]
        return f"Decoy{decoy_names[decoy_type]}({host})"
    else:  # 132-144
        return f"Restore({action_idx - 132})"


def print_obs_comparison(jax_obs, cyborg_obs):
    """Print side-by-side observation comparison."""
    host_names = ["Defender", "Enterprise0", "Enterprise1", "Enterprise2",
                  "Op_Host0", "Op_Host1", "Op_Host2", "Op_Server0",
                  "User0", "User1", "User2", "User3", "User4"]

    print("\n" + "="*80)
    print("OBSERVATION COMPARISON (JaxMARL vs CybORG)")
    print("="*80)
    print(f"{'Host':<15} {'JaxMARL':<30} {'CybORG':<30} {'Match'}")
    print("-"*80)

    mismatches = []
    for i, host in enumerate(host_names):
        jax_act = jax_obs[i*4:i*4+2]
        jax_comp = jax_obs[i*4+2:i*4+4]
        cyb_act = cyborg_obs[i*4:i*4+2]
        cyb_comp = cyborg_obs[i*4+2:i*4+4]

        jax_str = f"act={list(jax_act)} comp={list(jax_comp)}"
        cyb_str = f"act={list(cyb_act)} comp={list(cyb_comp)}"

        match = np.allclose(jax_obs[i*4:(i+1)*4], cyborg_obs[i*4:(i+1)*4])
        mark = "✓" if match else "✗ MISMATCH"

        if not match:
            mismatches.append(host)

        print(f"{host:<15} {jax_str:<30} {cyb_str:<30} {mark}")

    return mismatches


def run_cyborg_with_random_blue(num_steps=10, seed=42):
    """Run CybORG with random Blue actions."""
    path = '/home/paulhax/src/cyber/cage-challenge-2/CybORG/CybORG/Shared/Scenarios/Scenario2.yaml'
    cyborg = CybORG(path, 'sim', agents={'Red': B_lineAgent})
    env = BlueTableWrapper(cyborg, output_mode='vector')

    result = env.reset()
    obs = result.observation
    print(f"\n[CybORG] Initial obs shape: {obs.shape}")

    observations = [obs.copy()]
    actions = []

    for step in range(num_steps):
        np.random.seed(seed + step)
        action = np.random.randint(0, 145)
        actions.append(action)

        result = env.step(agent='Blue', action=action)
        obs = result.observation
        observations.append(obs.copy())

        print(f"[CybORG] Step {step}: action={get_action_type_name(action)}, reward={result.reward:.2f}")

    return observations, actions


def run_jaxmarl_with_scripted_red(num_steps=10, seed=42):
    """Run JaxMARL with scripted B_lineAgent red."""
    env = HeuristicRedCAGE()
    key = jax.random.PRNGKey(seed)

    key, reset_key = jax.random.split(key)
    obs, state = env.reset(reset_key)

    blue_obs = obs['blue']
    print(f"\n[JaxMARL] Initial obs shape: {blue_obs.shape}")

    observations = [np.array(blue_obs)]
    actions = []

    for step in range(num_steps):
        np.random.seed(seed + step)
        blue_action = np.random.randint(0, 145)
        actions.append(blue_action)

        key, step_key = jax.random.split(key)
        action_dict = {'blue': jnp.array(blue_action)}  # Red is scripted in HeuristicRedCAGE
        obs, state, reward, done, info = env.step(step_key, state, action_dict)

        observations.append(np.array(obs['blue']))

        print(f"[JaxMARL] Step {step}: action={get_action_type_name(blue_action)}, reward={reward['blue']:.2f}")

    return observations, actions


def compare_action_masks():
    """Compare JaxMARL action masks at initial state."""
    print("\n" + "="*80)
    print("ACTION MASK ANALYSIS (JaxMARL)")
    print("="*80)

    env = HeuristicRedCAGE()
    key = jax.random.PRNGKey(42)
    obs, state = env.reset(key)

    const = create_scenario2_const()
    jax_mask = get_blue_action_mask(state.state, const)

    print(f"\nAction validity at initial state:")
    print(f"  Sleep (0): {bool(jax_mask[0])}")
    print(f"  Monitor (1): {bool(jax_mask[1])}")
    print(f"  Analyse (2-14): all valid = {all(jax_mask[2:15])}, count = {sum(jax_mask[2:15])}")
    print(f"  Remove (15-27): valid count = {sum(jax_mask[15:28])}")

    if sum(jax_mask[15:28]) < 13:
        print(f"    Remove actions masked (expected: no activity detected yet):")
        for i in range(13):
            if not jax_mask[15+i]:
                host_names = ["Defender", "Enterprise0", "Enterprise1", "Enterprise2",
                              "Op_Host0", "Op_Host1", "Op_Host2", "Op_Server0",
                              "User0", "User1", "User2", "User3", "User4"]
                print(f"      Remove({host_names[i]}) = MASKED")

    print(f"  Decoy (28-131): valid count = {sum(jax_mask[28:132])}")
    print(f"  Restore (132-144): all valid = {all(jax_mask[132:145])}, count = {sum(jax_mask[132:145])}")


def test_policy_in_cyborg():
    """Test a trained JaxMARL policy in CybORG and analyze behavior."""
    print("\n" + "="*80)
    print("TESTING TRAINED POLICY IN CYBORG")
    print("="*80)

    # Find a checkpoint
    import glob
    checkpoints = glob.glob('/home/paulhax/src/cyber/jaxmarl-training/experiments/*/checkpoint_final.pkl')
    if not checkpoints:
        print("No checkpoints found!")
        return

    checkpoint_path = checkpoints[0]
    print(f"\nUsing checkpoint: {checkpoint_path}")

    # Load policy
    agent = JaxPolicyAgent(checkpoint_path)
    print(f"Policy: obs_dim={agent.obs_dim}, hidden_dim={agent.hidden_dim}, action_dim={agent.action_dim}")

    # Run in CybORG
    path = '/home/paulhax/src/cyber/cage-challenge-2/CybORG/CybORG/Shared/Scenarios/Scenario2.yaml'
    cyborg = CybORG(path, 'sim', agents={'Red': B_lineAgent})
    env = BlueTableWrapper(cyborg, output_mode='vector')

    result = env.reset()
    obs = result.observation
    agent.set_initial_values(None, obs)

    action_counts = {name: 0 for name in BLUE_ACTION_NAMES}
    total_reward = 0
    num_steps = 100

    print(f"\nRunning {num_steps} steps with trained policy...")

    for step in range(num_steps):
        action = agent.get_action(obs, None)

        # Categorize action
        if action == 0:
            action_type = "Sleep"
        elif action == 1:
            action_type = "Monitor"
        elif action < 15:
            action_type = "Analyse"
        elif action < 28:
            action_type = "Remove"
        elif action < 132:
            action_type = "Decoy"
        else:
            action_type = "Restore"
        action_counts[action_type] += 1

        result = env.step(agent='Blue', action=action)
        obs = result.observation
        total_reward += result.reward

        # Print every 20 steps
        if step % 20 == 0:
            print(f"  Step {step}: {get_action_type_name(action)}, reward={result.reward:.2f}")

    print(f"\nAction distribution over {num_steps} steps:")
    for name, count in action_counts.items():
        pct = 100 * count / num_steps
        print(f"  {name}: {count} ({pct:.1f}%)")

    print(f"\nTotal reward: {total_reward:.2f}")

    # Show action probabilities for initial state
    result = env.reset()
    obs = result.observation
    probs = agent.get_action_probabilities(obs)

    print(f"\nAction probabilities for initial state (top 10):")
    top_actions = np.argsort(probs)[-10:][::-1]
    for action in top_actions:
        print(f"  {get_action_type_name(action)}: {probs[action]:.4f}")


def main():
    print("="*80)
    print("JAXMARL vs CYBORG TRANSFER DEBUGGING")
    print("="*80)

    # Analyze action masks
    compare_action_masks()

    # Run parallel episodes and compare observations
    print("\n" + "="*80)
    print("PARALLEL EPISODE COMPARISON")
    print("="*80)

    cyborg_obs, cyborg_actions = run_cyborg_with_random_blue(num_steps=5)
    jaxmarl_obs, jaxmarl_actions = run_jaxmarl_with_scripted_red(num_steps=5)

    print("\nINITIAL STATE COMPARISON:")
    mismatches = print_obs_comparison(jaxmarl_obs[0], cyborg_obs[0])
    if mismatches:
        print(f"\n*** WARNING: Initial observations differ at hosts: {mismatches}")

    # Test trained policy
    test_policy_in_cyborg()


if __name__ == "__main__":
    main()
