"""Policy factories for comparison tests.

Provides configurable policies for Blue and Red agents to enable
systematic testing of different scenarios.
"""

from typing import Callable, List, Optional
import numpy as np

from jaxmarl.environments.cage.state import HOST_IDS
from jaxmarl.environments.cage.actions import (
    BLUE_SLEEP, BLUE_MONITOR, BLUE_REMOVE_START, BLUE_RESTORE_START,
    BLUE_ANALYSE_START, BLUE_DECOY_START, NUM_DECOY_TYPES,
    RED_SLEEP, RED_DISCOVER_SUBNET_START, RED_SCAN_HOST_START,
    RED_EXPLOIT_START, RED_PRIVESC_START, RED_IMPACT_START,
    NUM_BLUE_ACTIONS, NUM_RED_ACTIONS, NUM_HOSTS, NUM_EXPLOITS,
)
from tests.cage.differential.state_comparator import StateSnapshot


def sleep_policy(state: StateSnapshot, step: int) -> int:
    """Blue policy that always sleeps."""
    return BLUE_SLEEP


def monitor_policy(state: StateSnapshot, step: int) -> int:
    """Blue policy that always monitors."""
    return BLUE_MONITOR


def meander_policy_factory(seed: int) -> Callable[[StateSnapshot, int], int]:
    """Create a meander (random exploration) red policy.

    Simulates random exploration through the network, choosing
    valid-looking actions randomly.
    """
    rng = np.random.RandomState(seed)

    def policy(state: StateSnapshot, step: int) -> int:
        action_type = rng.choice(['discover', 'scan', 'exploit', 'privesc', 'impact', 'sleep'],
                                  p=[0.1, 0.2, 0.3, 0.2, 0.1, 0.1])

        if action_type == 'sleep':
            return RED_SLEEP
        elif action_type == 'discover':
            subnet = rng.randint(0, 3)
            return RED_DISCOVER_SUBNET_START + subnet
        elif action_type == 'scan':
            host_idx = rng.randint(0, NUM_HOSTS)
            return RED_SCAN_HOST_START + host_idx
        elif action_type == 'exploit':
            exploit_type = rng.randint(0, 8)
            host_idx = rng.randint(0, NUM_HOSTS)
            return RED_EXPLOIT_START + host_idx * NUM_EXPLOITS + exploit_type
        elif action_type == 'privesc':
            host_idx = rng.randint(0, NUM_HOSTS)
            return RED_PRIVESC_START + host_idx
        else:  # impact
            host_idx = rng.randint(0, NUM_HOSTS)
            return RED_IMPACT_START + host_idx

    return policy


def random_blue_policy_factory(seed: int) -> Callable[[StateSnapshot, int], int]:
    """Create a random blue policy."""
    rng = np.random.RandomState(seed)

    def policy(state: StateSnapshot, step: int) -> int:
        return rng.randint(0, NUM_BLUE_ACTIONS)

    return policy


def random_red_policy_factory(seed: int) -> Callable[[StateSnapshot, int], int]:
    """Create a random red policy."""
    rng = np.random.RandomState(seed)

    def policy(state: StateSnapshot, step: int) -> int:
        return rng.randint(0, NUM_RED_ACTIONS)

    return policy


def react_remove_policy_with_timing(delay: int = 0) -> Callable[[StateSnapshot, int], int]:
    """Create reactive remove policy with configurable delay.

    Args:
        delay: Steps to wait after detection before acting
    """
    detection_time = {}

    def policy(state: StateSnapshot, step: int) -> int:
        for hostname, detected in state.host_activity_detected.items():
            if detected and hostname not in detection_time:
                detection_time[hostname] = step

        for hostname, detect_step in sorted(detection_time.items()):
            if step >= detect_step + delay:
                if state.host_activity_detected.get(hostname, False):
                    host_idx = HOST_IDS[hostname]
                    return BLUE_REMOVE_START + host_idx

        return BLUE_MONITOR

    return policy


def react_restore_policy_with_timing(delay: int = 0) -> Callable[[StateSnapshot, int], int]:
    """Create reactive restore policy with configurable delay.

    Args:
        delay: Steps to wait after detection before acting
    """
    detection_time = {}

    def policy(state: StateSnapshot, step: int) -> int:
        for hostname, detected in state.host_activity_detected.items():
            if detected and hostname not in detection_time:
                detection_time[hostname] = step

        for hostname, detect_step in sorted(detection_time.items()):
            if step >= detect_step + delay:
                if state.host_activity_detected.get(hostname, False):
                    host_idx = HOST_IDS[hostname]
                    return BLUE_RESTORE_START + host_idx

        return BLUE_MONITOR

    return policy


def decoy_defense_policy(placement: List[tuple]) -> Callable[[StateSnapshot, int], int]:
    """Create strategic decoy deployment policy.

    Args:
        placement: List of (step, hostname, decoy_type) tuples
    """
    placement_dict = {step: (host, decoy) for step, host, decoy in placement}

    def policy(state: StateSnapshot, step: int) -> int:
        if step in placement_dict:
            hostname, decoy_type = placement_dict[step]
            host_idx = HOST_IDS[hostname]
            return BLUE_DECOY_START + host_idx * NUM_DECOY_TYPES + decoy_type
        return BLUE_MONITOR

    return policy


def scripted_action_sequence(
    actions: List[int],
    default_action: int = 0
) -> Callable[[StateSnapshot, int], int]:
    """Create policy from explicit action sequence.

    Args:
        actions: List of action indices to execute in order
        default_action: Action to use after sequence exhausted
    """
    def policy(state: StateSnapshot, step: int) -> int:
        if step < len(actions):
            return actions[step]
        return default_action

    return policy


def scripted_blue_policy_factory(actions: List[int]) -> Callable[[StateSnapshot, int], int]:
    """Create scripted blue policy from action list."""
    return scripted_action_sequence(actions, BLUE_SLEEP)


def scripted_red_policy_factory(actions: List[int]) -> Callable[[StateSnapshot, int], int]:
    """Create scripted red policy from action list."""
    return scripted_action_sequence(actions, RED_SLEEP)


def analyse_then_remove_policy() -> Callable[[StateSnapshot, int], int]:
    """Blue policy that analyses suspicious hosts then removes detected threats."""
    analysed_hosts = set()

    def policy(state: StateSnapshot, step: int) -> int:
        for hostname, detected in state.host_activity_detected.items():
            if detected:
                host_idx = HOST_IDS[hostname]
                if hostname not in analysed_hosts:
                    analysed_hosts.add(hostname)
                    return BLUE_ANALYSE_START + host_idx
                return BLUE_REMOVE_START + host_idx

        return BLUE_MONITOR

    return policy


def targeted_remove_policy(target_host: str, at_step: int) -> Callable[[StateSnapshot, int], int]:
    """Blue policy that removes specific host at specific step."""
    def policy(state: StateSnapshot, step: int) -> int:
        if step == at_step:
            return BLUE_REMOVE_START + HOST_IDS[target_host]
        return BLUE_MONITOR

    return policy


def targeted_restore_policy(target_host: str, at_step: int) -> Callable[[StateSnapshot, int], int]:
    """Blue policy that restores specific host at specific step."""
    def policy(state: StateSnapshot, step: int) -> int:
        if step == at_step:
            return BLUE_RESTORE_START + HOST_IDS[target_host]
        return BLUE_SLEEP

    return policy
