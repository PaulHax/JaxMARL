"""Core differential testing harness for CybORG <-> CAGE-JAX comparison.

This module provides the main DifferentialHarness class that runs both
environments in lockstep with identical actions and compares results.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Any
from pathlib import Path
import random
import importlib.util
import numpy as np

import jax
import jax.numpy as jnp

from jaxmarl.environments.cage import CageEnv
from jaxmarl.environments.cage.state import (
    HOST_IDS, CageState, CageConst,
    COMPROMISE_NONE, COMPROMISE_USER, COMPROMISE_PRIVILEGED,
)
from jaxmarl.environments.cage.actions import (
    BLUE_SLEEP, BLUE_MONITOR, NUM_BLUE_ACTIONS, NUM_RED_ACTIONS,
    get_red_action_offsets, get_blue_action_offsets,
)
from jaxmarl.environments.cage.scripted_agents import (
    bline_reset, bline_get_action, BLineState,
)
from jaxmarl.environments.cage.config import get_scenario, ScenarioConfig

from .action_translator import (
    cyborg_action_to_jax,
    jax_action_to_cyborg,
    describe_jax_blue_action,
    describe_jax_red_action,
)
from .state_comparator import (
    extract_cyborg_state,
    extract_jax_state,
    compare_states,
    states_match,
    format_state_comparison,
    StateSnapshot,
    StateDiff,
)


@dataclass
class StepResult:
    """Result of a single step in both environments."""
    step: int
    blue_action_jax: int
    red_action_jax: int
    blue_action_desc: str
    red_action_desc: str
    cyborg_state: StateSnapshot
    jax_state: StateSnapshot
    diffs: List[StateDiff]
    cyborg_action_success: bool = True
    jax_action_success: bool = True


@dataclass
class TestResult:
    """Result of a complete differential test run."""
    seed: int
    max_steps: int
    steps_completed: int
    step_results: List[StepResult] = field(default_factory=list)
    total_diffs: int = 0
    error_diffs: int = 0
    warning_diffs: int = 0
    passed: bool = True
    failure_reason: Optional[str] = None

    @property
    def cyborg_total_reward(self) -> Tuple[float, float]:
        """Get total CybORG rewards (blue, red)."""
        if not self.step_results:
            return 0.0, 0.0
        last = self.step_results[-1]
        return last.cyborg_state.reward_blue, last.cyborg_state.reward_red

    @property
    def jax_total_reward(self) -> Tuple[float, float]:
        """Get total JAX rewards (blue, red)."""
        if not self.step_results:
            return 0.0, 0.0
        last = self.step_results[-1]
        return last.jax_state.reward_blue, last.jax_state.reward_red

    def summary(self) -> str:
        """Get a summary of the test result."""
        status = "PASSED" if self.passed else "FAILED"
        lines = [
            f"Test Result: {status}",
            f"  Seed: {self.seed}",
            f"  Steps: {self.steps_completed}/{self.max_steps}",
            f"  Total diffs: {self.total_diffs} (errors: {self.error_diffs}, warnings: {self.warning_diffs})",
            f"  CybORG rewards: blue={self.cyborg_total_reward[0]:.2f}, red={self.cyborg_total_reward[1]:.2f}",
            f"  JAX rewards:    blue={self.jax_total_reward[0]:.2f}, red={self.jax_total_reward[1]:.2f}",
        ]
        if self.failure_reason:
            lines.append(f"  Failure: {self.failure_reason}")
        return "\n".join(lines)


def _get_cyborg_scenario_path(scenario_name: str = "Scenario2.yaml") -> Path:
    """Get path to CybORG scenario file.

    Searches for the scenario in standard CybORG paths:
    - CybORG/Shared/Scenarios/
    - CybORG/Shared/Scenarios/scalability_experiments/
    """
    spec = importlib.util.find_spec("CybORG")
    if spec is None or spec.origin is None:
        raise ImportError("CybORG package not found")
    cyborg_path = Path(spec.origin).parent
    scenarios_dir = cyborg_path / "Shared" / "Scenarios"

    scenario_path = scenarios_dir / scenario_name
    if scenario_path.exists():
        return scenario_path

    scalability_path = scenarios_dir / "scalability_experiments" / scenario_name
    if scalability_path.exists():
        return scalability_path

    raise FileNotFoundError(
        f"Scenario file not found in {scenarios_dir} or {scenarios_dir / 'scalability_experiments'}: {scenario_name}"
    )


class DifferentialHarness:
    """Differential testing harness for CybORG and CAGE-JAX comparison.

    This class runs both environments in lockstep, executing identical
    actions and comparing state after each step.
    """

    def __init__(
        self,
        seed: int = 42,
        max_steps: int = 100,
        scenario: str = "Scenario2.yaml",
        check_rewards: bool = True,
        check_obs: bool = False,
        verbose: bool = False,
        sync_detection_rng: bool = False,
    ):
        """Initialize the differential harness.

        Args:
            seed: Random seed for reproducibility
            max_steps: Maximum steps per episode
            scenario: CybORG scenario file name
            check_rewards: Whether to compare rewards
            check_obs: Whether to compare observations
            verbose: Whether to print verbose output
        """
        self.seed = seed
        self.max_steps = max_steps
        self.scenario = scenario
        self.check_rewards = check_rewards
        self.check_obs = check_obs
        self.verbose = verbose
        self.sync_detection_rng = sync_detection_rng

        self.cyborg_env = None
        self.jax_env = None
        self.jax_state = None
        self.jax_key = None
        self.step_count = 0
        self.red_known_ips = {}  # Track IPs Red has discovered
        self.config = self._load_scenario_config()

    def _load_scenario_config(self) -> ScenarioConfig:
        """Load scenario configuration for JAX environment."""
        scenario_name = self.scenario
        if scenario_name.endswith('.yaml'):
            scenario_name = scenario_name[:-5]
        return get_scenario(scenario_name)

    def _create_cyborg_env(self):
        """Create CybORG environment.

        For scenarios using AlignedReward calculator (e.g., hosts_2), passes
        the required agent_reward_params for the ResilienceMetric.
        """
        from CybORG import CybORG
        scenario_name = self.scenario
        if not scenario_name.endswith('.yaml'):
            scenario_name = scenario_name + '.yaml'
        scenario_path = _get_cyborg_scenario_path(scenario_name)

        agent_rewards = None
        if 'hosts_' in scenario_name:
            agent_rewards = {
                'Blue': {
                    'align_attribute': 'C',
                    'alignment_metric': 'ResilienceMetric',
                    'alpha': 1.0,
                }
            }

        env = CybORG(scenario_file=str(scenario_path), environment='sim', agent_rewards=agent_rewards)
        env.set_seed(self.seed)
        return env

    def _create_jax_env(self) -> CageEnv:
        """Create CAGE-JAX environment.

        For hosts_* scenarios, enables resilience_gamma to match CybORG's
        AlignedRewardCalculator with ResilienceMetric.
        """
        resilience_gamma = 0.0
        if 'hosts_' in self.scenario:
            resilience_gamma = 1.0

        return CageEnv(
            config=self.config,
            max_steps=self.max_steps,
            resilience_gamma=resilience_gamma,
        )

    def reset(self) -> Tuple[StateSnapshot, StateSnapshot]:
        """Reset both environments.

        Returns:
            Tuple of (cyborg_state, jax_state) snapshots
        """
        self.cyborg_env = self._create_cyborg_env()
        self.jax_env = self._create_jax_env()

        self.cyborg_env.reset()

        self.jax_key = jax.random.PRNGKey(self.seed)
        obs, self.jax_state = self.jax_env.reset(self.jax_key)

        if self.sync_detection_rng:
            # Use Python RNG sequence for exploit detection to align with CybORG
            rng = random.Random(self.seed)
            seq_len = int(self.jax_env.const.max_steps)
            detection_seq = jnp.array([rng.random() for _ in range(seq_len)], dtype=jnp.float32)
            self.jax_state = self.jax_state.replace(
                exploit_detection_randoms=detection_seq,
                exploit_detection_index=jnp.array(0, dtype=jnp.int32),
                use_exploit_detection_randoms=jnp.array(True),
            )

        self.step_count = 0
        self.red_known_ips = {}
        self.red_discovered_hosts = set()
        self.red_scanned_hosts = set()
        self._update_known_ips_from_observation()

        cyborg_state = extract_cyborg_state(
            self.cyborg_env,
            config=self.config,
            include_obs=self.check_obs,
            discovered_hosts_override=self.red_discovered_hosts,
            scanned_hosts_override=self.red_scanned_hosts,
        )
        jax_state = extract_jax_state(self.jax_state, self.jax_env.const, config=self.config, include_obs=self.check_obs)

        return cyborg_state, jax_state

    def _update_known_ips_from_observation(self):
        """Extract IPs that Red has discovered from CybORG observation."""
        obs = self.cyborg_env.get_observation('Red')
        if obs is None:
            return

        ip_map = self.cyborg_env.get_ip_map()
        for hostname, ip in ip_map.items():
            ip_str = str(ip)
            if ip_str in obs:
                self.red_known_ips[hostname] = ip
                self.red_discovered_hosts.add(hostname)
                host_info = obs.get(ip_str, {})
                if isinstance(host_info, dict) and ('Processes' in host_info or 'Services' in host_info):
                    self.red_scanned_hosts.add(hostname)

    def step(
        self,
        blue_action_jax: int,
        red_action_jax: int,
    ) -> StepResult:
        """Execute one step in both environments.

        Args:
            blue_action_jax: Blue agent action as JAX index
            red_action_jax: Red agent action as JAX index

        Returns:
            StepResult with state comparison
        """
        blue_cyborg = jax_action_to_cyborg(blue_action_jax, self.cyborg_env, 'Blue', config=self.config)
        red_cyborg = jax_action_to_cyborg(red_action_jax, self.cyborg_env, 'Red', self.red_known_ips, config=self.config)

        # Step Blue and capture action cost (before Red's step overwrites it)
        self.cyborg_env.step('Blue', blue_cyborg)
        blue_action_cost = blue_cyborg.cost if hasattr(blue_cyborg, 'cost') else 0

        # Step Red to complete the timestep
        self.cyborg_env.step('Red', red_cyborg)

        self._update_known_ips_from_observation()

        # Get state-based rewards after both steps, then add Blue's action cost
        cyborg_blue_reward = self.cyborg_env.get_rewards()['Blue'] + blue_action_cost
        cyborg_red_reward = self.cyborg_env.get_rewards()['Red']

        self.jax_key, subkey = jax.random.split(self.jax_key)
        actions = {
            'blue': jnp.array(blue_action_jax),
            'red': jnp.array(red_action_jax),
        }
        obs, self.jax_state, rewards, dones, info = self.jax_env.step_env(
            subkey, self.jax_state, actions
        )

        self.step_count += 1

        cyborg_state = extract_cyborg_state(
            self.cyborg_env,
            config=self.config,
            include_obs=self.check_obs,
            discovered_hosts_override=self.red_discovered_hosts,
            scanned_hosts_override=self.red_scanned_hosts,
        )
        cyborg_state.reward_blue = cyborg_blue_reward
        cyborg_state.reward_red = cyborg_red_reward

        jax_state_snapshot = extract_jax_state(
            self.jax_state, self.jax_env.const, config=self.config, include_obs=self.check_obs
        )

        jax_state_snapshot.reward_blue = float(rewards['blue'])
        jax_state_snapshot.reward_red = float(rewards['red'])

        diffs = compare_states(
            cyborg_state, jax_state_snapshot,
            config=self.config,
            check_rewards=self.check_rewards,
            check_obs=self.check_obs,
        )

        result = StepResult(
            step=self.step_count,
            blue_action_jax=blue_action_jax,
            red_action_jax=red_action_jax,
            blue_action_desc=describe_jax_blue_action(blue_action_jax, self.config),
            red_action_desc=describe_jax_red_action(red_action_jax, self.config, self.jax_env.const),
            cyborg_state=cyborg_state,
            jax_state=jax_state_snapshot,
            diffs=diffs,
            jax_action_success=bool(self.jax_state.last_red_action_success),
        )

        if self.verbose and diffs:
            print(f"\nStep {self.step_count}: {result.red_action_desc}")
            for diff in diffs:
                print(f"  {diff}")

        return result

    def run_episode(
        self,
        blue_policy: Callable[[StateSnapshot, int], int],
        red_policy: Callable[[StateSnapshot, int], int],
    ) -> TestResult:
        """Run a complete episode with given policies.

        Args:
            blue_policy: Function (state, step) -> blue_action_jax
            red_policy: Function (state, step) -> red_action_jax

        Returns:
            TestResult with complete episode data
        """
        cyborg_state, jax_state = self.reset()

        result = TestResult(
            seed=self.seed,
            max_steps=self.max_steps,
            steps_completed=0,
        )

        for step in range(self.max_steps):
            try:
                blue_action = blue_policy(jax_state, step)
                red_action = red_policy(jax_state, step)

                step_result = self.step(blue_action, red_action)
                result.step_results.append(step_result)
                result.steps_completed = step + 1

                for diff in step_result.diffs:
                    result.total_diffs += 1
                    if diff.severity == 'error':
                        result.error_diffs += 1
                    else:
                        result.warning_diffs += 1

                jax_state = step_result.jax_state

            except Exception as e:
                result.passed = False
                result.failure_reason = f"Exception at step {step}: {e}"
                break

        if result.error_diffs > 0:
            result.passed = False
            result.failure_reason = f"Found {result.error_diffs} error-level differences"

        return result

    def run_bline_episode(
        self,
        blue_policy: Callable[[StateSnapshot, int], int],
        use_jax_bline: bool = True,
    ) -> TestResult:
        """Run episode with B_lineAgent for red.

        Args:
            blue_policy: Function (state, step) -> blue_action_jax
            use_jax_bline: If True, use JAX B_lineAgent; if False, use CybORG's

        Returns:
            TestResult with complete episode data
        """
        if use_jax_bline:
            bline_state = bline_reset()
            num_hosts = self.config.num_hosts
            num_red_actions = get_red_action_offsets(num_hosts)[-1]

            def red_policy(state: StateSnapshot, step: int) -> int:
                nonlocal bline_state

                red_obs = jnp.array([1.0 if self.jax_state.last_red_action_success else 0.0])
                red_obs = jnp.concatenate([red_obs, jnp.zeros(num_hosts * 3)])

                action_mask = jnp.ones(num_red_actions, dtype=jnp.bool_)

                key = jax.random.PRNGKey(self.seed + step)
                action, bline_state = bline_get_action(
                    bline_state, red_obs, action_mask,
                    self.jax_env.const, key,
                    self.jax_state.host_services
                )
                return int(action)

            return self.run_episode(blue_policy, red_policy)

        else:
            from CybORG.Agents.SimpleAgents.B_line import B_lineAgent

            cyborg_agent = B_lineAgent()
            red_action_jax = 0

            def red_policy(state: StateSnapshot, step: int) -> int:
                nonlocal red_action_jax

                obs = self.cyborg_env.get_observation('Red')
                action_space = self.cyborg_env.get_action_space('Red')
                cyborg_action = cyborg_agent.get_action(obs, action_space)

                red_action_jax = cyborg_action_to_jax(
                    cyborg_action, self.cyborg_env, 'Red', config=self.config
                )
                return red_action_jax

            cyborg_state, jax_state = self.reset()

            result = TestResult(
                seed=self.seed,
                max_steps=self.max_steps,
                steps_completed=0,
            )

            for step in range(self.max_steps):
                try:
                    blue_action = blue_policy(jax_state, step)

                    obs = self.cyborg_env.get_observation('Red')
                    action_space = self.cyborg_env.get_action_space('Red')
                    cyborg_red_action = cyborg_agent.get_action(obs, action_space)

                    blue_cyborg = jax_action_to_cyborg(blue_action, self.cyborg_env, 'Blue', config=self.config)

                    # Step Blue and capture action cost (before Red's step overwrites it)
                    self.cyborg_env.step('Blue', blue_cyborg)
                    blue_action_cost = blue_cyborg.cost if hasattr(blue_cyborg, 'cost') else 0

                    # Step Red to complete the timestep
                    self.cyborg_env.step('Red', cyborg_red_action)
                    self._update_known_ips_from_observation()

                    # Get state-based rewards after both steps, then add Blue's action cost
                    cyborg_blue_reward = self.cyborg_env.get_rewards()['Blue'] + blue_action_cost
                    cyborg_red_reward = self.cyborg_env.get_rewards()['Red']

                    red_action_jax = cyborg_action_to_jax(
                        cyborg_red_action, self.cyborg_env, 'Red', config=self.config
                    )

                    self.jax_key, subkey = jax.random.split(self.jax_key)
                    actions = {
                        'blue': jnp.array(blue_action),
                        'red': jnp.array(red_action_jax),
                    }
                    obs_jax, self.jax_state, rewards, dones, info = self.jax_env.step_env(
                        subkey, self.jax_state, actions
                    )

                    self.step_count += 1

                    cyborg_state_snap = extract_cyborg_state(
                        self.cyborg_env,
                        config=self.config,
                        include_obs=self.check_obs,
                        discovered_hosts_override=self.red_discovered_hosts,
                        scanned_hosts_override=self.red_scanned_hosts,
                    )
                    cyborg_state_snap.reward_blue = cyborg_blue_reward
                    cyborg_state_snap.reward_red = cyborg_red_reward

                    jax_state_snap = extract_jax_state(
                        self.jax_state, self.jax_env.const, config=self.config, include_obs=self.check_obs
                    )
                    jax_state_snap.reward_blue = float(rewards['blue'])
                    jax_state_snap.reward_red = float(rewards['red'])

                    diffs = compare_states(
                        cyborg_state_snap, jax_state_snap,
                        config=self.config,
                        check_rewards=self.check_rewards,
                        check_obs=self.check_obs,
                    )

                    step_result = StepResult(
                        step=step + 1,
                        blue_action_jax=blue_action,
                        red_action_jax=red_action_jax,
                        blue_action_desc=describe_jax_blue_action(blue_action, self.config),
                        red_action_desc=describe_jax_red_action(red_action_jax, self.config, self.jax_env.const),
                        cyborg_state=cyborg_state_snap,
                        jax_state=jax_state_snap,
                        diffs=diffs,
                        jax_action_success=bool(self.jax_state.last_red_action_success),
                    )

                    result.step_results.append(step_result)
                    result.steps_completed = step + 1

                    for diff in diffs:
                        result.total_diffs += 1
                        if diff.severity == 'error':
                            result.error_diffs += 1
                        else:
                            result.warning_diffs += 1

                    jax_state = jax_state_snap

                    if self.verbose:
                        print(f"Step {step}: {step_result.red_action_desc}")
                        if diffs:
                            for diff in diffs:
                                print(f"  {diff}")

                except Exception as e:
                    result.passed = False
                    result.failure_reason = f"Exception at step {step}: {e}"
                    import traceback
                    traceback.print_exc()
                    break

            if result.error_diffs > 0:
                result.passed = False
                result.failure_reason = f"Found {result.error_diffs} error-level differences"

            return result

    def run_meander_episode(
        self,
        blue_policy: Callable[[StateSnapshot, int], int],
    ) -> TestResult:
        """Run episode with CybORG's RedMeanderAgent for red.

        Uses CybORG's Meander agent and translates actions to JAX.

        Args:
            blue_policy: Function (state, step) -> blue_action_jax

        Returns:
            TestResult with complete episode data
        """
        from CybORG.Agents.SimpleAgents.Meander import RedMeanderAgent

        cyborg_agent = RedMeanderAgent()

        cyborg_state, jax_state = self.reset()

        result = TestResult(
            seed=self.seed,
            max_steps=self.max_steps,
            steps_completed=0,
        )

        for step in range(self.max_steps):
            try:
                blue_action = blue_policy(jax_state, step)

                obs = self.cyborg_env.get_observation('Red')
                action_space = self.cyborg_env.get_action_space('Red')
                cyborg_red_action = cyborg_agent.get_action(obs, action_space)

                blue_cyborg = jax_action_to_cyborg(blue_action, self.cyborg_env, 'Blue', config=self.config)

                self.cyborg_env.step('Blue', blue_cyborg)
                blue_action_cost = blue_cyborg.cost if hasattr(blue_cyborg, 'cost') else 0

                self.cyborg_env.step('Red', cyborg_red_action)
                self._update_known_ips_from_observation()

                cyborg_blue_reward = self.cyborg_env.get_rewards()['Blue'] + blue_action_cost
                cyborg_red_reward = self.cyborg_env.get_rewards()['Red']

                red_action_jax = cyborg_action_to_jax(
                    cyborg_red_action, self.cyborg_env, 'Red', config=self.config
                )

                self.jax_key, subkey = jax.random.split(self.jax_key)
                actions = {
                    'blue': jnp.array(blue_action),
                    'red': jnp.array(red_action_jax),
                }
                obs_jax, self.jax_state, rewards, dones, info = self.jax_env.step_env(
                    subkey, self.jax_state, actions
                )

                self.step_count += 1

                cyborg_state_snap = extract_cyborg_state(
                    self.cyborg_env,
                    config=self.config,
                    include_obs=self.check_obs,
                    discovered_hosts_override=self.red_discovered_hosts,
                    scanned_hosts_override=self.red_scanned_hosts,
                )
                cyborg_state_snap.reward_blue = cyborg_blue_reward
                cyborg_state_snap.reward_red = cyborg_red_reward

                jax_state_snap = extract_jax_state(
                    self.jax_state, self.jax_env.const, config=self.config, include_obs=self.check_obs
                )
                jax_state_snap.reward_blue = float(rewards['blue'])
                jax_state_snap.reward_red = float(rewards['red'])

                diffs = compare_states(
                    cyborg_state_snap, jax_state_snap,
                    config=self.config,
                    check_rewards=self.check_rewards,
                    check_obs=self.check_obs,
                )

                step_result = StepResult(
                    step=step + 1,
                    blue_action_jax=blue_action,
                    red_action_jax=red_action_jax,
                    blue_action_desc=describe_jax_blue_action(blue_action, self.config),
                    red_action_desc=describe_jax_red_action(red_action_jax, self.config, self.jax_env.const),
                    cyborg_state=cyborg_state_snap,
                    jax_state=jax_state_snap,
                    diffs=diffs,
                    jax_action_success=bool(self.jax_state.last_red_action_success),
                )

                result.step_results.append(step_result)
                result.steps_completed = step + 1

                for diff in diffs:
                    result.total_diffs += 1
                    if diff.severity == 'error':
                        result.error_diffs += 1
                    else:
                        result.warning_diffs += 1

                jax_state = jax_state_snap

                if self.verbose:
                    print(f"Step {step}: {step_result.red_action_desc}")
                    if diffs:
                        for diff in diffs:
                            print(f"  {diff}")

            except Exception as e:
                result.passed = False
                result.failure_reason = f"Exception at step {step}: {e}"
                import traceback
                traceback.print_exc()
                break

        if result.error_diffs > 0:
            result.passed = False
            result.failure_reason = f"Found {result.error_diffs} error-level differences"

        return result


def sleep_policy(state: StateSnapshot, step: int) -> int:
    """Blue policy that always sleeps."""
    return BLUE_SLEEP


def monitor_policy(state: StateSnapshot, step: int) -> int:
    """Blue policy that always monitors."""
    return BLUE_MONITOR


def reactive_remove_policy(state: StateSnapshot, step: int) -> int:
    """Blue policy that monitors and removes detected threats."""
    from jaxmarl.environments.cage.actions import BLUE_REMOVE_START

    for hostname, detected in state.host_activity_detected.items():
        if detected:
            host_idx = HOST_IDS[hostname]
            return BLUE_REMOVE_START + host_idx

    return BLUE_MONITOR


def reactive_restore_policy(state: StateSnapshot, step: int) -> int:
    """Blue policy that monitors and restores compromised hosts."""
    from jaxmarl.environments.cage.actions import BLUE_RESTORE_START

    for hostname, detected in state.host_activity_detected.items():
        if detected:
            host_idx = HOST_IDS[hostname]
            return BLUE_RESTORE_START + host_idx

    return BLUE_MONITOR


def random_blue_policy_factory(seed: int) -> Callable[[StateSnapshot, int], int]:
    """Create a random blue policy with given seed."""
    rng = np.random.RandomState(seed)

    def policy(state: StateSnapshot, step: int) -> int:
        return rng.randint(0, NUM_BLUE_ACTIONS)

    return policy


def random_red_policy_factory(seed: int) -> Callable[[StateSnapshot, int], int]:
    """Create a random red policy with given seed."""
    rng = np.random.RandomState(seed)

    def policy(state: StateSnapshot, step: int) -> int:
        return rng.randint(0, NUM_RED_ACTIONS)

    return policy


def scripted_red_policy_factory(actions: List[int]) -> Callable[[StateSnapshot, int], int]:
    """Create a scripted red policy from action list."""
    def policy(state: StateSnapshot, step: int) -> int:
        if step < len(actions):
            return actions[step]
        return 0

    return policy


def scripted_blue_policy_factory(actions: List[int]) -> Callable[[StateSnapshot, int], int]:
    """Create a scripted blue policy from action list."""
    from jaxmarl.environments.cage.actions import BLUE_SLEEP

    def policy(state: StateSnapshot, step: int) -> int:
        if step < len(actions):
            return actions[step]
        return BLUE_SLEEP

    return policy


def conditional_blue_policy_factory(
    condition_fn: Callable[[StateSnapshot, int], bool],
    action_if_true: int,
    action_if_false: int,
) -> Callable[[StateSnapshot, int], int]:
    """Create a conditional blue policy.

    Args:
        condition_fn: Function (state, step) -> bool
        action_if_true: Action to take if condition is true
        action_if_false: Action to take if condition is false

    Returns:
        Policy function
    """
    def policy(state: StateSnapshot, step: int) -> int:
        if condition_fn(state, step):
            return action_if_true
        return action_if_false

    return policy


def is_cyborg_available() -> bool:
    """Check if CybORG is installed."""
    spec = importlib.util.find_spec("CybORG")
    return spec is not None


class JaxOnlyHarness:
    """JAX-only testing harness when CybORG is not available.

    This harness runs only the CAGE-JAX environment without comparison
    to CybORG. Useful for testing JAX behavior in isolation.
    """

    def __init__(
        self,
        seed: int = 42,
        max_steps: int = 100,
        verbose: bool = False,
    ):
        """Initialize the JAX-only harness.

        Args:
            seed: Random seed for reproducibility
            max_steps: Maximum steps per episode
            verbose: Whether to print verbose output
        """
        self.seed = seed
        self.max_steps = max_steps
        self.verbose = verbose

        self.jax_env = None
        self.jax_state = None
        self.jax_key = None
        self.step_count = 0

    def _create_jax_env(self) -> CageEnv:
        """Create CAGE-JAX environment."""
        return CageEnv(max_steps=self.max_steps)

    def reset(self) -> StateSnapshot:
        """Reset the JAX environment.

        Returns:
            JAX state snapshot
        """
        self.jax_env = self._create_jax_env()
        self.jax_key = jax.random.PRNGKey(self.seed)
        obs, self.jax_state = self.jax_env.reset(self.jax_key)
        self.step_count = 0

        return extract_jax_state(self.jax_state, self.jax_env.const, include_obs=True)

    def step(
        self,
        blue_action_jax: int,
        red_action_jax: int,
    ) -> Tuple[StateSnapshot, dict]:
        """Execute one step in JAX environment.

        Args:
            blue_action_jax: Blue agent action as JAX index
            red_action_jax: Red agent action as JAX index

        Returns:
            Tuple of (state_snapshot, rewards_dict)
        """
        self.jax_key, subkey = jax.random.split(self.jax_key)
        actions = {
            'blue': jnp.array(blue_action_jax),
            'red': jnp.array(red_action_jax),
        }
        obs, self.jax_state, rewards, dones, info = self.jax_env.step_env(
            subkey, self.jax_state, actions
        )

        self.step_count += 1

        state_snapshot = extract_jax_state(
            self.jax_state, self.jax_env.const, include_obs=True
        )
        state_snapshot.reward_blue = float(rewards['blue'])
        state_snapshot.reward_red = float(rewards['red'])

        return state_snapshot, {
            'blue': float(rewards['blue']),
            'red': float(rewards['red']),
            'done': bool(dones['__all__']),
        }

    def run_episode(
        self,
        blue_policy: Callable[[StateSnapshot, int], int],
        red_policy: Callable[[StateSnapshot, int], int],
    ) -> TestResult:
        """Run a complete episode with given policies.

        Args:
            blue_policy: Function (state, step) -> blue_action_jax
            red_policy: Function (state, step) -> red_action_jax

        Returns:
            TestResult with episode data (no CybORG comparison)
        """
        jax_state = self.reset()

        result = TestResult(
            seed=self.seed,
            max_steps=self.max_steps,
            steps_completed=0,
        )

        for step in range(self.max_steps):
            try:
                blue_action = blue_policy(jax_state, step)
                red_action = red_policy(jax_state, step)

                jax_state, rewards = self.step(blue_action, red_action)

                step_result = StepResult(
                    step=step + 1,
                    blue_action_jax=blue_action,
                    red_action_jax=red_action,
                    blue_action_desc=describe_jax_blue_action(blue_action),
                    red_action_desc=describe_jax_red_action(red_action),
                    cyborg_state=StateSnapshot(),
                    jax_state=jax_state,
                    diffs=[],
                    jax_action_success=bool(self.jax_state.last_red_action_success),
                )

                result.step_results.append(step_result)
                result.steps_completed = step + 1

                if self.verbose:
                    print(f"Step {step + 1}: {step_result.red_action_desc}")

                if rewards.get('done', False):
                    break

            except Exception as e:
                result.passed = False
                result.failure_reason = f"Exception at step {step}: {e}"
                import traceback
                traceback.print_exc()
                break

        return result

    def run_bline_episode(
        self,
        blue_policy: Callable[[StateSnapshot, int], int],
    ) -> TestResult:
        """Run episode with JAX B_lineAgent for red.

        Args:
            blue_policy: Function (state, step) -> blue_action_jax

        Returns:
            TestResult with episode data
        """
        bline_state = bline_reset()

        def red_policy(state: StateSnapshot, step: int) -> int:
            nonlocal bline_state

            red_obs = jnp.array([1.0 if self.jax_state.last_red_action_success else 0.0])
            red_obs = jnp.concatenate([red_obs, jnp.zeros(len(HOST_IDS) * 3)])

            action_mask = jnp.ones(NUM_RED_ACTIONS, dtype=jnp.bool_)

            key = jax.random.PRNGKey(self.seed + step)
            action, bline_state = bline_get_action(
                bline_state, red_obs, action_mask,
                self.jax_env.const, key,
                self.jax_state.host_services
            )
            return int(action)

        return self.run_episode(blue_policy, red_policy)
