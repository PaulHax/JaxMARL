"""Shared CybORG evaluation utilities for JaxMARL training scripts."""

import sys
from pathlib import Path

from jaxmarl.environments.cage.actions import BLUE_ACTION_NAMES

CYBORG_AVAILABLE = False


def setup_cyborg_eval(cyborg_path: str):
    """Set up CybORG imports for evaluation."""
    global CYBORG_AVAILABLE
    cyborg_path = Path(cyborg_path)
    if cyborg_path.exists():
        sys.path.insert(0, str(cyborg_path))
        try:
            from CybORG.Agents.SimpleAgents.JaxPolicyAgent import JaxPolicyAgent
            from CybORG.Agents import B_lineAgent
            from cage_experiment import CAGEExperiment
            from CybORG.AlignmentMetric.resilience_measure import ResilienceMetric
            CYBORG_AVAILABLE = True
            return True
        except ImportError as e:
            print(f"Warning: Could not import CybORG components: {e}")
            return False
    return False


def evaluate_in_cyborg(checkpoint_path: str, cyborg_path: str, episodes: int = 10,
                       steps: int = 100, seed: int = 42, track_actions: bool = True):
    """Evaluate a JaxMARL checkpoint in CybORG and return CIA metrics.

    Returns dict with: confidentiality, integrity, availability, resilience, reward,
    and optionally action distribution percentages.
    """
    if not CYBORG_AVAILABLE:
        return None

    cyborg_dir = Path(cyborg_path)
    sys.path.insert(0, str(cyborg_dir))

    from CybORG.Agents.SimpleAgents.JaxPolicyAgent import JaxPolicyAgent
    from CybORG.Agents import B_lineAgent
    from cage_experiment import CAGEExperiment
    from CybORG.AlignmentMetric.resilience_measure import ResilienceMetric

    agent = JaxPolicyAgent(checkpoint_path)
    metric = ResilienceMetric()

    cage = CAGEExperiment(
        agent,
        red_agent=B_lineAgent,
        scenario="Scenario2",
        seed=seed,
        metric=metric,
        experiment_export_dir="/tmp/jax_eval",
        use_wrapper=True
    )
    agent.set_env = lambda env: None

    action_counts = {name: 0 for name in BLUE_ACTION_NAMES}
    total_actions = 0

    if track_actions and hasattr(agent, 'last_actions'):
        original_get_action = agent.get_action

        def tracked_get_action(*args, **kwargs):
            nonlocal total_actions
            action = original_get_action(*args, **kwargs)
            if hasattr(agent, 'last_action_type'):
                action_type = agent.last_action_type
                if action_type < len(BLUE_ACTION_NAMES):
                    action_counts[BLUE_ACTION_NAMES[action_type]] += 1
                    total_actions += 1
            return action

        agent.get_action = tracked_get_action

    results = cage.run_experiment(
        episodes=episodes,
        steps=steps,
        plot_export_path="eval.png",
        verbose=False
    )

    result_dict = {
        "confidentiality": results[0],
        "integrity": results[1],
        "availability": results[2],
        "resilience": results[3],
        "reward": results[4],
        "confidentiality_std": results[5],
        "integrity_std": results[6],
        "availability_std": results[7],
        "resilience_std": results[8],
        "reward_std": results[9],
    }

    if track_actions and total_actions > 0:
        for name in BLUE_ACTION_NAMES:
            result_dict[f"pct_{name.lower()}"] = action_counts[name] / total_actions * 100

    return result_dict


def log_cyborg_eval_results(cia_results, mlflow_module=None, step=None, prefix="final", verbose=True):
    """Log CybORG evaluation results to MLflow and optionally print summary.

    Args:
        cia_results: Dict from evaluate_in_cyborg()
        mlflow_module: The mlflow module (pass mlflow directly)
        step: Step number for MLflow logging
        prefix: Metric prefix for MLflow ("final" or "eval")
        verbose: Whether to print detailed results
    """
    if not cia_results:
        return

    if mlflow_module and step is not None:
        eval_metrics = {
            f"{prefix}/confidentiality": cia_results["confidentiality"],
            f"{prefix}/integrity": cia_results["integrity"],
            f"{prefix}/availability": cia_results["availability"],
            f"{prefix}/resilience": cia_results["resilience"],
            f"{prefix}/cyborg_reward": cia_results["reward"],
        }
        for name in BLUE_ACTION_NAMES:
            pct_key = f"pct_{name.lower()}"
            if pct_key in cia_results:
                eval_metrics[f"{prefix}/{pct_key}"] = cia_results[pct_key]

        mlflow_module.log_metrics(eval_metrics, step=step)

    if verbose:
        print(f"  Confidentiality: {cia_results['confidentiality']:.3f} ± {cia_results['confidentiality_std']:.3f}")
        print(f"  Integrity:       {cia_results['integrity']:.3f} ± {cia_results['integrity_std']:.3f}")
        print(f"  Availability:    {cia_results['availability']:.3f} ± {cia_results['availability_std']:.3f}")
        print(f"  Resilience:      {cia_results['resilience']:.3f} ± {cia_results['resilience_std']:.3f}")
        print(f"  CybORG Reward:   {cia_results['reward']:.2f} ± {cia_results['reward_std']:.2f}")

        action_pcts = [f"{name}={cia_results.get(f'pct_{name.lower()}', 0):.1f}%"
                      for name in BLUE_ACTION_NAMES]
        print(f"  Actions: {', '.join(action_pcts)}")


def format_cyborg_eval_summary(cia_results):
    """Return a one-line summary of CybORG evaluation results."""
    if not cia_results:
        return ""
    return (f"C={cia_results['confidentiality']:.2f} "
            f"I={cia_results['integrity']:.2f} "
            f"A={cia_results['availability']:.2f} "
            f"R={cia_results['resilience']:.2f} "
            f"Reward={cia_results['reward']:.1f}")
