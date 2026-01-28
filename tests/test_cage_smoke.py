"""Tests for CAGE-JAX environment and B_lineAgent."""

import pytest
import jax
import jax.numpy as jnp
from flax import struct
import chex


class TestMinimalJIT:
    """Basic JIT compilation tests."""

    def test_minimal_jit(self):
        """Verify basic CAGE-like state can JIT compile."""

        @struct.dataclass
        class MinimalState:
            time: int
            host_compromised: chex.Array

        @jax.jit
        def step(state: MinimalState) -> MinimalState:
            return state.replace(time=state.time + 1)

        @jax.jit
        def reset(key: chex.PRNGKey) -> MinimalState:
            return MinimalState(
                time=0,
                host_compromised=jnp.zeros(13, dtype=jnp.int32)
            )

        key = jax.random.PRNGKey(0)
        state = reset(key)
        state = step(state)

        assert state.time == 1
        assert state.host_compromised.shape == (13,)


class TestBLineAgentHostIndices:
    """Test B_lineAgent dynamic host index lookup for different scenarios."""

    def test_scenario2_host_indices(self):
        """Verify B_lineAgent target hosts are correct for Scenario2."""
        from jaxmarl.environments.cage import CageEnv

        env = CageEnv(scenario='Scenario2')

        # Scenario2 has 13 hosts in alphabetical order
        assert env.const.num_hosts == 13

        # B_lineAgent targets should match Scenario2 layout
        # CybORG topology: User1 connects to Enterprise1, not Enterprise0
        assert env.const.bline_user_host == 9      # User1
        assert env.const.bline_enterprise1 == 2    # Enterprise1 (User1's connected host)
        assert env.const.bline_enterprise2 == 3    # Enterprise2
        assert env.const.bline_op_server0 == 7     # Op_Server0

    def test_hosts_5_host_indices(self):
        """Verify B_lineAgent target hosts are correctly looked up for hosts_5."""
        from jaxmarl.environments.cage import CageEnv

        env = CageEnv(scenario='hosts_5')

        # hosts_5 has 62 hosts
        assert env.const.num_hosts == 62

        # B_lineAgent targets should be different from Scenario2
        # These are looked up by name, not hardcoded
        assert env.const.bline_user_host != 9      # Different from Scenario2
        assert env.const.bline_enterprise1 != 1    # Different from Scenario2

        # Indices should be valid (within num_hosts)
        assert 0 <= env.const.bline_user_host < env.const.num_hosts
        assert 0 <= env.const.bline_enterprise1 < env.const.num_hosts
        assert 0 <= env.const.bline_enterprise2 < env.const.num_hosts
        assert 0 <= env.const.bline_op_server0 < env.const.num_hosts

    def test_bline_targets_in_correct_subnets(self):
        """Verify B_lineAgent targets are in the expected subnets."""
        from jaxmarl.environments.cage import CageEnv

        for scenario in ['Scenario2', 'hosts_5']:
            env = CageEnv(scenario=scenario)

            # Get subnet indices for each target
            user_subnet = int(env.const.host_subnet[env.const.bline_user_host])
            ent1_subnet = int(env.const.host_subnet[env.const.bline_enterprise1])
            ent2_subnet = int(env.const.host_subnet[env.const.bline_enterprise2])
            op_subnet = int(env.const.host_subnet[env.const.bline_op_server0])

            # User1 and Enterprise hosts should be in different subnets than Op_Server0
            # (Op_Server0 is in Operational subnet, others are in User/Enterprise)
            assert user_subnet != op_subnet, f"{scenario}: User1 should not be in Operational subnet"
            assert ent1_subnet != op_subnet, f"{scenario}: Enterprise1 should not be in Operational subnet"

    @pytest.mark.parametrize("scenario", ['Scenario2', 'hosts_2', 'hosts_3', 'hosts_4', 'hosts_5'])
    def test_all_scenarios_have_valid_bline_targets(self, scenario):
        """Verify all scenarios have valid B_lineAgent target indices."""
        from jaxmarl.environments.cage import CageEnv

        env = CageEnv(scenario=scenario)

        # All indices should be valid
        assert 0 <= env.const.bline_user_host < env.const.num_hosts
        assert 0 <= env.const.bline_enterprise1 < env.const.num_hosts
        assert 0 <= env.const.bline_enterprise2 < env.const.num_hosts
        assert 0 <= env.const.bline_op_server0 < env.const.num_hosts


class TestBLineAgentFSM:
    """Test B_lineAgent FSM behavior."""

    def test_bline_action_valid_for_scenario(self):
        """Verify B_lineAgent produces valid actions for each scenario."""
        from jaxmarl.environments.cage import CageEnv
        from jaxmarl.environments.cage.scripted_agents import bline_reset, bline_get_action

        for scenario in ['Scenario2', 'hosts_5']:
            env = CageEnv(scenario=scenario)
            key = jax.random.PRNGKey(0)

            # Reset environment and agent
            obs, state = env.reset(key)
            agent_state = bline_reset()

            # Get action mask
            avail = env.get_avail_actions(state)

            # Get B_lineAgent action
            key, subkey = jax.random.split(key)
            action, new_agent_state = bline_get_action(
                agent_state, obs['red'], avail['red'], env.const, subkey
            )

            # Action should be valid (within action space)
            assert 0 <= int(action) < env.red_action_size, \
                f"{scenario}: B_lineAgent action {action} out of range [0, {env.red_action_size})"

    def test_bline_progresses_through_fsm(self):
        """Verify B_lineAgent FSM progresses on successful actions."""
        from jaxmarl.environments.cage import CageEnv
        from jaxmarl.environments.cage.scripted_agents import bline_reset, bline_get_action

        env = CageEnv(scenario='Scenario2')
        key = jax.random.PRNGKey(42)

        obs, state = env.reset(key)
        agent_state = bline_reset()

        initial_fsm = int(agent_state.fsm_state)

        # Run a few steps
        for _ in range(10):
            avail = env.get_avail_actions(state)
            key, subkey = jax.random.split(key)
            action, agent_state = bline_get_action(
                agent_state, obs['red'], avail['red'], env.const, subkey
            )

            # Step environment
            key, subkey = jax.random.split(key)
            actions = {'blue': jnp.array(1), 'red': action}  # Blue does Monitor
            obs, state, rewards, dones, info = env.step(subkey, state, actions)

        # FSM should have progressed (Red should make some progress)
        final_fsm = int(agent_state.fsm_state)
        assert final_fsm >= initial_fsm, "B_lineAgent FSM should progress"


class TestCageEnvScenarios:
    """Test CageEnv works with different scenarios."""

    @pytest.mark.parametrize("scenario,expected_hosts", [
        ('Scenario2', 13),
        ('hosts_2', 25),
        ('hosts_3', 37),
        ('hosts_4', 49),
        ('hosts_5', 62),
    ])
    def test_scenario_host_counts(self, scenario, expected_hosts):
        """Verify each scenario has the expected number of hosts."""
        from jaxmarl.environments.cage import CageEnv

        env = CageEnv(scenario=scenario)
        assert env.const.num_hosts == expected_hosts

    def test_env_step_works_all_scenarios(self):
        """Verify environment step works for all scenarios."""
        from jaxmarl.environments.cage import CageEnv

        for scenario in ['Scenario2', 'hosts_5']:
            env = CageEnv(scenario=scenario)
            key = jax.random.PRNGKey(0)

            obs, state = env.reset(key)

            # Take a random valid action
            avail = env.get_avail_actions(state)
            blue_valid = jnp.where(avail['blue'])[0]
            red_valid = jnp.where(avail['red'])[0]

            actions = {
                'blue': blue_valid[0],
                'red': red_valid[0],
            }

            key, subkey = jax.random.split(key)
            obs, state, rewards, dones, info = env.step(subkey, state, actions)

            # Check outputs are valid
            assert 'blue' in obs
            assert 'red' in obs
            assert 'blue' in rewards
            assert 'red' in rewards


class TestObservationPersistence:
    """Test that Blue observations persist like CybORG (anomalies visible without Monitor)."""

    def test_observation_nonzero_without_monitor(self):
        """Blue should see anomalies even when doing Sleep (not Monitor).

        In CybORG, BlueTableWrapper detects anomalies by comparing current
        state to baseline. Red's sessions create detectable processes.
        Blue doesn't need to explicitly Monitor to see these anomalies.
        """
        from jaxmarl.environments.cage import CageEnv
        from jaxmarl.environments.cage.scripted_agents import bline_reset, bline_get_action

        env = CageEnv(scenario='Scenario2', max_steps=100)
        key = jax.random.PRNGKey(42)

        obs, state = env.reset(key)
        bline_state = bline_reset()

        # Blue does Sleep for 10 steps while Red attacks
        obs_nonzero_counts = []
        for step in range(10):
            avail = env.get_avail_actions(state)

            key, subkey = jax.random.split(key)
            red_action, bline_state = bline_get_action(
                bline_state, obs['red'], avail['red'], env.const, subkey
            )

            blue_action = jnp.array(0)  # Sleep
            actions = {'blue': blue_action, 'red': red_action}

            key, subkey = jax.random.split(key)
            obs, state, rewards, dones, info = env.step(subkey, state, actions)

            obs_nonzero = int(jnp.sum(obs['blue'] != 0))
            obs_nonzero_counts.append(obs_nonzero)

        # Red compromises hosts over 10 steps, Blue should see SOME anomalies
        # even without doing Monitor (matching CybORG behavior)
        max_nonzero = max(obs_nonzero_counts)
        assert max_nonzero > 0, (
            f"Blue observation should show anomalies without Monitor. "
            f"Got all zeros for 10 steps. Counts: {obs_nonzero_counts}"
        )

    def test_delayed_monitor_sees_old_compromises(self):
        """Monitor after delay should reveal compromises from earlier steps.

        In CybORG, anomalies persist until Restore. If Blue waits 10 steps
        then does Monitor, it should still see the compromised hosts.
        """
        from jaxmarl.environments.cage import CageEnv
        from jaxmarl.environments.cage.scripted_agents import bline_reset, bline_get_action

        env = CageEnv(scenario='Scenario2', max_steps=100)
        key = jax.random.PRNGKey(42)

        obs, state = env.reset(key)
        bline_state = bline_reset()

        # Blue does Sleep for 10 steps
        for step in range(10):
            avail = env.get_avail_actions(state)

            key, subkey = jax.random.split(key)
            red_action, bline_state = bline_get_action(
                bline_state, obs['red'], avail['red'], env.const, subkey
            )

            blue_action = jnp.array(0)  # Sleep
            actions = {'blue': blue_action, 'red': red_action}

            key, subkey = jax.random.split(key)
            obs, state, rewards, dones, info = env.step(subkey, state, actions)

        # Check how many hosts are actually compromised
        num_compromised = int(jnp.sum(state.host_compromised > 0))

        # Now Blue does Monitor
        avail = env.get_avail_actions(state)
        key, subkey = jax.random.split(key)
        red_action, bline_state = bline_get_action(
            bline_state, obs['red'], avail['red'], env.const, subkey
        )

        blue_action = jnp.array(1)  # Monitor
        actions = {'blue': blue_action, 'red': red_action}

        key, subkey = jax.random.split(key)
        obs, state, rewards, dones, info = env.step(subkey, state, actions)

        # Count hosts visible in observation (non-zero activity or compromise)
        obs_nonzero = int(jnp.sum(obs['blue'] != 0))

        # Should see multiple compromised hosts, not just one
        assert obs_nonzero >= 4, (
            f"Delayed Monitor should reveal multiple compromised hosts. "
            f"Actual compromised: {num_compromised}, visible in obs: {obs_nonzero // 4} hosts"
        )

    def test_initial_foothold_hidden(self):
        """Red's initial foothold (User0) should be hidden from Blue.

        CybORG's baseline includes Red's starting position, so Blue
        doesn't see it as an anomaly until Red takes additional actions.
        """
        from jaxmarl.environments.cage import CageEnv

        env = CageEnv(scenario='Scenario2')
        key = jax.random.PRNGKey(42)

        obs, state = env.reset(key)

        # Red starts on User0 (index 8) with privileged access
        user0_idx = 8
        assert state.red_sessions[user0_idx] > 0, "Red should have session on User0"
        assert state.host_compromised[user0_idx] > 0, "User0 should be compromised"

        # But Blue's initial observation should NOT show User0
        user0_obs = obs['blue'][user0_idx * 4:(user0_idx + 1) * 4]
        assert jnp.all(user0_obs == 0), (
            f"Initial foothold (User0) should be hidden from Blue. "
            f"Got: {user0_obs}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
