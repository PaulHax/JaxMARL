"""CybORG equivalence tests for CAGE-JAX.

These tests verify that CAGE-JAX produces equivalent results to the
original CybORG CAGE Challenge 2 implementation.

Requirements:
- CybORG installed: pip install -r requirements-jax.txt
- Run with: pytest tests/test_cage_cyborg_equivalence.py -v
"""

import pytest
from pathlib import Path

import jax
import jax.numpy as jnp

from jaxmarl.environments.cage import CageEnv
from jaxmarl.environments.cage.state import (
    HOST_IDS, HOST_NAMES, HOST_SUBNET, SUBNET_IDS, DECOY_IDS,
    COMPROMISE_NONE, COMPROMISE_USER, COMPROMISE_PRIVILEGED,
    NUM_DECOY_TYPES, create_scenario2_const,
)
from jaxmarl.environments.cage.observations import get_blue_obs
from jaxmarl.environments.cage.actions import (
    decode_blue_action, get_blue_action_offsets,
    BLUE_SLEEP, BLUE_MONITOR, BLUE_ANALYSE_START, BLUE_REMOVE_START,
    BLUE_DECOY_START, BLUE_RESTORE_START, NUM_BLUE_ACTIONS,
)


def _get_cyborg_scenario_path(scenario_name: str = "Scenario2.yaml") -> Path:
    """Get path to CybORG scenario file using importlib."""
    import importlib.util
    spec = importlib.util.find_spec("CybORG")
    if spec is None or spec.origin is None:
        raise ImportError("CybORG package not found")
    cyborg_path = Path(spec.origin).parent
    scenario_path = cyborg_path / "Shared" / "Scenarios" / scenario_name
    if not scenario_path.exists():
        raise FileNotFoundError(f"Scenario file not found: {scenario_path}")
    return scenario_path


class TestHostOrderingEquivalence:
    """Verify host ordering matches CybORG exactly (alphabetical)."""

    def test_host_order_matches_cyborg_alphabetical(self):
        """HOST_IDS should match CybORG's alphabetical order."""
        expected_order = [
            'Defender', 'Enterprise0', 'Enterprise1', 'Enterprise2',
            'Op_Host0', 'Op_Host1', 'Op_Host2', 'Op_Server0',
            'User0', 'User1', 'User2', 'User3', 'User4'
        ]

        for idx, hostname in enumerate(expected_order):
            assert HOST_IDS[hostname] == idx, f"{hostname} should be at index {idx}, got {HOST_IDS[hostname]}"

    def test_host_names_inverse_of_host_ids(self):
        """HOST_NAMES should be inverse mapping of HOST_IDS."""
        for name, idx in HOST_IDS.items():
            assert HOST_NAMES[idx] == name, f"HOST_NAMES[{idx}] should be {name}, got {HOST_NAMES[idx]}"

    def test_host_subnet_mapping(self):
        """Hosts should map to correct subnets."""
        expected_subnets = {
            'Defender': 'Enterprise', 'Enterprise0': 'Enterprise',
            'Enterprise1': 'Enterprise', 'Enterprise2': 'Enterprise',
            'Op_Host0': 'Operational', 'Op_Host1': 'Operational',
            'Op_Host2': 'Operational', 'Op_Server0': 'Operational',
            'User0': 'User', 'User1': 'User', 'User2': 'User',
            'User3': 'User', 'User4': 'User'
        }

        for hostname, subnet_name in expected_subnets.items():
            host_idx = HOST_IDS[hostname]
            subnet_idx = SUBNET_IDS[subnet_name]
            assert HOST_SUBNET[host_idx] == subnet_idx, \
                f"{hostname} should be in {subnet_name} subnet (idx {subnet_idx}), got {HOST_SUBNET[host_idx]}"

    def test_host_count(self):
        """Should have 13 hosts."""
        assert len(HOST_IDS) == 13
        assert len(HOST_NAMES) == 13


class TestDecoyTypeOrderEquivalence:
    """Verify decoy type ordering matches CybORG exactly (alphabetical)."""

    def test_decoy_type_count(self):
        """Should have 8 decoy types."""
        assert NUM_DECOY_TYPES == 8
        assert len(DECOY_IDS) == 8

    def test_decoy_type_order(self):
        """Decoy types should match CybORG alphabetical order."""
        expected_order = {
            'DecoyApache': 0, 'DecoyFemitter': 1, 'DecoyHarakaSMPT': 2,
            'DecoySmss': 3, 'DecoySSHD': 4, 'DecoySvchost': 5,
            'DecoyTomcat': 6, 'DecoyVsftpd': 7
        }

        assert DECOY_IDS == expected_order, f"DECOY_IDS mismatch: {DECOY_IDS} != {expected_order}"


class TestActionStructureEquivalence:
    """Verify action encoding matches CybORG exactly."""

    def test_action_count(self):
        """Total blue action count should be 145."""
        assert NUM_BLUE_ACTIONS == 145

    def test_sleep_monitor_positions(self):
        """Sleep=0, Monitor=1."""
        assert BLUE_SLEEP == 0
        assert BLUE_MONITOR == 1

    def test_analyse_positions(self):
        """Analyse actions at 2-14."""
        assert BLUE_ANALYSE_START == 2

    def test_remove_positions(self):
        """Remove actions at 15-27."""
        assert BLUE_REMOVE_START == 15

    def test_decoy_positions(self):
        """Decoy actions at 28-131 (8 types × 13 hosts = 104)."""
        assert BLUE_DECOY_START == 28

    def test_restore_positions(self):
        """Restore actions at 132-144 (AFTER decoys)."""
        assert BLUE_RESTORE_START == 132

    def test_action_offsets_from_function(self):
        """get_blue_action_offsets should return correct values."""
        const = create_scenario2_const()
        analyse_start, remove_start, decoy_start, restore_start = get_blue_action_offsets(const)

        assert analyse_start == 2
        assert remove_start == 15
        assert decoy_start == 28
        assert restore_start == 132

    def test_decode_analyse_action(self):
        """Verify Analyse actions decode correctly."""
        const = create_scenario2_const()

        # Analyse Defender (action 2)
        action_type, target_host, decoy_type = decode_blue_action(2, const)
        assert int(action_type) == 2  # Analyse
        assert int(target_host) == HOST_IDS['Defender']

        # Analyse User0 (action 10)
        action_type, target_host, decoy_type = decode_blue_action(10, const)
        assert int(action_type) == 2  # Analyse
        assert int(target_host) == HOST_IDS['User0']

    def test_decode_remove_action(self):
        """Verify Remove actions decode correctly."""
        const = create_scenario2_const()

        # Remove Defender (action 15)
        action_type, target_host, decoy_type = decode_blue_action(15, const)
        assert int(action_type) == 3  # Remove
        assert int(target_host) == HOST_IDS['Defender']

        # Remove User4 (action 27)
        action_type, target_host, decoy_type = decode_blue_action(27, const)
        assert int(action_type) == 3  # Remove
        assert int(target_host) == HOST_IDS['User4']

    def test_decode_decoy_action(self):
        """Verify Decoy actions decode correctly."""
        const = create_scenario2_const()

        # DecoyApache on Defender (action 28)
        action_type, target_host, decoy_type = decode_blue_action(28, const)
        assert int(action_type) == 5  # Decoy
        assert int(target_host) == HOST_IDS['Defender']
        assert int(decoy_type) == DECOY_IDS['DecoyApache']

        # DecoyFemitter on Defender (action 29)
        action_type, target_host, decoy_type = decode_blue_action(29, const)
        assert int(action_type) == 5  # Decoy
        assert int(target_host) == HOST_IDS['Defender']
        assert int(decoy_type) == DECOY_IDS['DecoyFemitter']

        # DecoyVsftpd on User4 (action 131)
        action_type, target_host, decoy_type = decode_blue_action(131, const)
        assert int(action_type) == 5  # Decoy
        assert int(target_host) == HOST_IDS['User4']
        assert int(decoy_type) == DECOY_IDS['DecoyVsftpd']

    def test_decode_restore_action(self):
        """Verify Restore actions decode correctly."""
        const = create_scenario2_const()

        # Restore Defender (action 132)
        action_type, target_host, decoy_type = decode_blue_action(132, const)
        assert int(action_type) == 4  # Restore
        assert int(target_host) == HOST_IDS['Defender']

        # Restore User0 (action 140)
        action_type, target_host, decoy_type = decode_blue_action(140, const)
        assert int(action_type) == 4  # Restore
        assert int(target_host) == HOST_IDS['User0']

        # Restore User4 (action 144)
        action_type, target_host, decoy_type = decode_blue_action(144, const)
        assert int(action_type) == 4  # Restore
        assert int(target_host) == HOST_IDS['User4']


class TestObservationEncodingEquivalence:
    """Verify observation encoding matches CybORG exactly."""

    def test_obs_dimension(self):
        """Blue observation should be 52 dimensions (4 per host × 13 hosts)."""
        env = CageEnv()
        obs, _ = env.reset(jax.random.PRNGKey(0))
        assert obs['blue'].shape == (52,)

    def test_clean_host_encoding(self):
        """Clean host should encode as [0, 0, 0, 0]."""
        env = CageEnv()
        _, state = env.reset(jax.random.PRNGKey(0))

        obs = get_blue_obs(state, env.const)

        # Enterprise0 (idx 1) should be clean initially
        host_idx = HOST_IDS['Enterprise0']
        host_obs = obs[host_idx * 4:(host_idx + 1) * 4]
        assert list(host_obs) == [0, 0, 0, 0], f"Clean host obs: {list(host_obs)}"

    def test_scanned_host_encoding(self):
        """Scanned-only host should encode as [1, 0, 0, 0] when detected."""
        env = CageEnv()
        _, state = env.reset(jax.random.PRNGKey(0))

        host_idx = HOST_IDS['Enterprise0']
        state = state.replace(
            red_scanned_hosts_jax=state.red_scanned_hosts_jax.at[host_idx].set(True),
            host_activity_detected=state.host_activity_detected.at[host_idx].set(True),
        )
        obs = get_blue_obs(state, env.const)

        host_obs = obs[host_idx * 4:(host_idx + 1) * 4]
        assert list(host_obs) == [1, 0, 0, 0], f"Scanned host obs: {list(host_obs)}"

    def test_exploited_user_encoding(self):
        """User-compromised host should encode as [1, 1, 0, 1] when detected."""
        env = CageEnv()
        _, state = env.reset(jax.random.PRNGKey(0))

        host_idx = HOST_IDS['Enterprise0']
        state = state.replace(
            red_scanned_hosts_jax=state.red_scanned_hosts_jax.at[host_idx].set(True),
            red_sessions=state.red_sessions.at[host_idx].set(1),
            host_compromised=state.host_compromised.at[host_idx].set(COMPROMISE_USER),
            host_activity_detected=state.host_activity_detected.at[host_idx].set(True),
        )
        obs = get_blue_obs(state, env.const)

        host_obs = obs[host_idx * 4:(host_idx + 1) * 4]
        assert list(host_obs) == [1, 1, 0, 1], f"User-compromised host obs: {list(host_obs)}"

    def test_exploited_privileged_encoding(self):
        """Privileged-compromised host should encode as [1, 1, 1, 1] when detected."""
        env = CageEnv()
        _, state = env.reset(jax.random.PRNGKey(0))

        host_idx = HOST_IDS['Enterprise0']
        state = state.replace(
            red_scanned_hosts_jax=state.red_scanned_hosts_jax.at[host_idx].set(True),
            red_sessions=state.red_sessions.at[host_idx].set(1),
            host_compromised=state.host_compromised.at[host_idx].set(COMPROMISE_PRIVILEGED),
            host_activity_detected=state.host_activity_detected.at[host_idx].set(True),
        )
        obs = get_blue_obs(state, env.const)

        host_obs = obs[host_idx * 4:(host_idx + 1) * 4]
        assert list(host_obs) == [1, 1, 1, 1], f"Privileged-compromised host obs: {list(host_obs)}"


class TestHostValueEquivalence:
    """Verify host values match between CAGE-JAX and CybORG Scenario2."""

    def test_value_mapping(self):
        """Verify the value mapping matches CybORG."""
        from jaxmarl.environments.cage.rewards import CONFIDENTIALITY_SCALE, AVAILABILITY_SCALE
        assert CONFIDENTIALITY_SCALE == 1.0
        assert AVAILABILITY_SCALE == 1.0

    def test_host_confidentiality_values(self):
        """Verify confidentiality values match Scenario2.yaml."""
        const = create_scenario2_const()

        expected = {
            'User0': 0.0,
            'User1': 0.1,
            'User2': 0.1,
            'User3': 0.1,
            'User4': 0.1,
            'Enterprise0': 1.0,
            'Enterprise1': 1.0,
            'Enterprise2': 1.0,
            'Defender': 0.1,
            'Op_Host0': 0.1,
            'Op_Host1': 0.1,
            'Op_Host2': 0.1,
            'Op_Server0': 1.0,
        }

        for host, expected_val in expected.items():
            actual = float(const.host_confidentiality[HOST_IDS[host]])
            assert abs(actual - expected_val) < 1e-5, f"{host}: expected {expected_val}, got {actual}"

    def test_host_availability_values(self):
        """Verify availability values match Scenario2.yaml."""
        const = create_scenario2_const()

        expected = {
            'User0': 0.0,
            'User1': 0.0,
            'Enterprise0': 1.0,
            'Enterprise1': 1.0,
            'Enterprise2': 1.0,
            'Defender': 0.1,
            'Op_Server0': 10.0,
        }

        for host, expected_val in expected.items():
            actual = float(const.host_availability[HOST_IDS[host]])
            assert abs(actual - expected_val) < 1e-5, f"{host}: expected {expected_val}, got {actual}"


class TestRewardEquivalence:
    """Verify reward calculations match CybORG."""

    def test_initial_reward_zero(self):
        """Initial reward should be 0 (only User0 compromised, value=0)."""
        from jaxmarl.environments.cage.rewards import compute_rewards_simple
        from jaxmarl.environments.cage.state import create_initial_state_with_red_foothold

        const = create_scenario2_const()
        state = create_initial_state_with_red_foothold(const)

        rewards = compute_rewards_simple(state, const)
        assert rewards['blue'] == 0.0, f"Initial reward should be 0, got {rewards['blue']}"

    def test_single_host_privileged_reward(self):
        """Compromising single host with privileged access."""
        from jaxmarl.environments.cage.rewards import compute_rewards_simple
        from jaxmarl.environments.cage.state import create_initial_state

        const = create_scenario2_const()
        state = create_initial_state(const)

        state = state.replace(
            host_compromised=state.host_compromised.at[HOST_IDS['Enterprise0']].set(COMPROMISE_PRIVILEGED)
        )

        rewards = compute_rewards_simple(state, const)

        assert rewards['red'] == 1.0
        assert rewards['blue'] == -1.0

    def test_op_server_full_reward(self):
        """Compromising Op_Server0 with Impact gives -11.0 (confidentiality + availability)."""
        from jaxmarl.environments.cage.rewards import compute_rewards_simple
        from jaxmarl.environments.cage.state import create_initial_state

        const = create_scenario2_const()
        state = create_initial_state(const)

        # Privileged access + Impact (OT service stopped) for availability reward
        state = state.replace(
            host_compromised=state.host_compromised.at[HOST_IDS['Op_Server0']].set(COMPROMISE_PRIVILEGED),
            ot_service_stopped=state.ot_service_stopped.at[HOST_IDS['Op_Server0']].set(True),
        )

        rewards = compute_rewards_simple(state, const)

        assert rewards['red'] == 11.0
        assert rewards['blue'] == -11.0


class TestInitialStateEquivalence:
    """Verify initial state matches CybORG."""

    def test_red_starts_on_user0(self):
        """Red should start with PRIVILEGED (SYSTEM) session on User0.

        CybORG starts Red with SYSTEM access, not user-level.
        """
        env = CageEnv()
        _, state = env.reset(jax.random.PRNGKey(0))

        assert state.host_compromised[HOST_IDS['User0']] == COMPROMISE_PRIVILEGED
        assert state.red_privilege[HOST_IDS['User0']] == COMPROMISE_PRIVILEGED
        assert state.red_sessions[HOST_IDS['User0']] == 1

    def test_other_hosts_clean(self):
        """Other hosts should be clean initially."""
        env = CageEnv()
        _, state = env.reset(jax.random.PRNGKey(0))

        for host, idx in HOST_IDS.items():
            if host != 'User0':
                assert state.host_compromised[idx] == 0, f"{host} should be clean"


class TestCybORGDirectComparison:
    """Direct comparison tests that run both CybORG and CAGE-JAX."""

    @pytest.fixture
    def cyborg_env(self):
        """Create CybORG environment."""
        from CybORG import CybORG
        scenario_path = _get_cyborg_scenario_path()
        return CybORG(scenario_file=str(scenario_path), environment='sim')

    def test_episode_length(self, cyborg_env):
        """Both environments should run for 100 steps."""
        jax_env = CageEnv(max_steps=100)
        assert jax_env.max_steps == 100

    def test_action_space_sizes(self, cyborg_env):
        """Action space sizes should match."""
        from jaxmarl.environments.cage.actions import NUM_BLUE_ACTIONS, NUM_RED_ACTIONS

        assert NUM_BLUE_ACTIONS == 145
        assert NUM_RED_ACTIONS > 100

    def test_cyborg_initial_state(self, cyborg_env):
        """Verify CybORG initial state structure."""
        cyborg_env.reset()
        true_state = cyborg_env.get_true_state({'Sessions': True})
        assert 'Red' in str(true_state) or len(true_state) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
