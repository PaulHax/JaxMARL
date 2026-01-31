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
                       steps: int = 100, seed: int = 42, track_actions: bool = True,
                       export_dir: str = None, red_agent_name: str = "bline"):
    """Evaluate a JaxMARL checkpoint in CybORG and return CIA metrics.

    Args:
        red_agent_name: "bline" or "meander"

    Returns dict with: confidentiality, integrity, availability, resilience, reward,
    optionally action distribution percentages, and trajectory_dir path.
    """
    if not CYBORG_AVAILABLE:
        return None

    cyborg_dir = Path(cyborg_path)
    sys.path.insert(0, str(cyborg_dir))

    from CybORG.Agents.SimpleAgents.JaxPolicyAgent import JaxPolicyAgent
    from CybORG.Agents import B_lineAgent
    from CybORG.Agents.SimpleAgents.Meander import RedMeanderAgent
    from cage_experiment import CAGEExperiment
    from CybORG.AlignmentMetric.resilience_measure import ResilienceMetric

    red_agent_class = RedMeanderAgent if red_agent_name == "meander" else B_lineAgent

    agent = JaxPolicyAgent(checkpoint_path)
    metric = ResilienceMetric()

    if export_dir is None:
        export_dir = "/tmp/jax_eval"

    cage = CAGEExperiment(
        agent,
        red_agent=red_agent_class,
        scenario="Scenario2",
        seed=seed,
        metric=metric,
        experiment_export_dir=export_dir,
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
        plot_export_path=f"eval_{red_agent_name}.png",
        verbose=False
    )

    trajectory_dir = Path(export_dir) / "trajectories" / f"{agent}-{red_agent_class.__name__}"

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
        "trajectory_dir": str(trajectory_dir),
        "red_agent": red_agent_name,
    }

    if track_actions and total_actions > 0:
        for name in BLUE_ACTION_NAMES:
            result_dict[f"pct_{name.lower()}"] = action_counts[name] / total_actions * 100

    return result_dict


def run_final_cyborg_eval(checkpoint_path: str, cyborg_path: str, mlflow_module,
                          total_steps: int, export_dir: str, episodes: int = 3,
                          steps: int = 100, seed: int = 42):
    """Run CybORG evaluation against both B_line and Meander, log trajectories to MLflow.

    This is the default post-training evaluation that logs trajectory JSONs as artifacts.
    """
    if not setup_cyborg_eval(cyborg_path):
        print("Warning: CybORG not available, skipping final evaluation")
        return

    print("\n" + "=" * 60)
    print("Running CybORG evaluation (trajectories will be logged to MLflow)")
    print("=" * 60)

    for red_agent in ["bline", "meander"]:
        print(f"\nEvaluating against {red_agent.upper()}...")

        results = evaluate_in_cyborg(
            checkpoint_path, cyborg_path,
            episodes=episodes, steps=steps, seed=seed,
            export_dir=export_dir, red_agent_name=red_agent
        )

        if results:
            log_cyborg_eval_results(
                results, mlflow_module, total_steps,
                prefix=f"final/{red_agent}", verbose=True
            )


def log_cyborg_eval_results(cia_results, mlflow_module=None, step=None, prefix="final", verbose=True):
    """Log CybORG trajectory artifacts to MLflow and optionally print summary.

    Args:
        cia_results: Dict from evaluate_in_cyborg()
        mlflow_module: The mlflow module (pass mlflow directly)
        step: Step number (unused, kept for API compatibility)
        prefix: Unused, kept for API compatibility
        verbose: Whether to print detailed results
    """
    if not cia_results:
        return

    if mlflow_module:
        trajectory_dir = cia_results.get("trajectory_dir")
        red_agent = cia_results.get("red_agent", "unknown")
        if trajectory_dir:
            trajectory_path = Path(trajectory_dir)
            if trajectory_path.exists():
                artifact_subpath = f"trajectories/{red_agent}"
                mlflow_module.log_artifacts(str(trajectory_path), artifact_path=artifact_subpath)

                # Create HTML viewer that embeds cynex for each trajectory
                run_id = mlflow_module.active_run().info.run_id
                for traj_file in trajectory_path.glob("*.json"):
                    artifact_url = f"http://localhost:5000/get-artifact?path={artifact_subpath}/{traj_file.name}&run_uuid={run_id}"
                    cynex_url = f"http://localhost:5173?file={artifact_url}"
                    viewer_html = f'''<!DOCTYPE html>
<html>
<head><title>Trajectory Viewer - {traj_file.name}</title></head>
<body style="margin:0;padding:0;overflow:hidden;">
<iframe src="{cynex_url}"
        style="width:100%;height:100vh;border:none;"
        sandbox="allow-same-origin allow-scripts allow-popups">
</iframe>
<p style="position:absolute;bottom:10px;left:10px;font-family:sans-serif;font-size:12px;color:#666;">
  <a href="{cynex_url}" target="_blank">Open in new tab</a>
</p>
</body>
</html>'''
                    viewer_path = trajectory_path / f"view_{traj_file.stem}.html"
                    viewer_path.write_text(viewer_html)
                    mlflow_module.log_artifact(str(viewer_path), artifact_path=artifact_subpath)

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
