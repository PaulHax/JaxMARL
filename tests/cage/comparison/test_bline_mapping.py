"""Tests for B_lineAgent user-to-enterprise mapping parity."""

import inspect

import pytest

from tests.cage.differential.harness import is_cyborg_available
from jaxmarl.environments.cage.config import create_scenario2_config
from jaxmarl.environments.cage.state import build_const_from_config, HOST_IDS, HOST_NAMES


requires_cyborg = pytest.mark.skipif(
    not is_cyborg_available(),
    reason="CybORG not installed"
)


@requires_cyborg
def test_bline_user_to_enterprise_mapping_matches_cyborg(monkeypatch):
    """JAX B_line mapping should follow CybORG's User->Enterprise connectivity."""
    from CybORG import CybORG
    from CybORG.Agents.SimpleAgents.B_line import B_lineAgent
    import CybORG.Agents.SimpleAgents.B_line as bline_mod

    const = build_const_from_config(create_scenario2_config())

    def cyborg_enterprise_for_user(user_host: str) -> str:
        path = str(inspect.getfile(CybORG))
        path = path[:-10] + "/Shared/Scenarios/Scenario2.yaml"
        cyborg = CybORG(path, "sim", agents={"Red": B_lineAgent})
        cyborg.reset()
        agent = B_lineAgent()

        ip_map = cyborg.get_ip_map()
        target_ip = ip_map[user_host]

        def forced_choice(seq):
            for item in seq:
                if str(item) == str(target_ip):
                    return item
            return seq[0]

        monkeypatch.setattr(bline_mod.random, "choice", forced_choice)

        action_space = cyborg.get_action_space("Red")
        enterprise_host = None
        for _ in range(10):
            obs = cyborg.get_observation("Red")
            action = agent.get_action(obs, action_space)
            cyborg.step("Red", action)

            if action.__class__.__name__ != "DiscoverNetworkServices":
                continue

            params = action.get_params()
            ip = params.get("ip_address")
            if ip is None:
                continue

            for name, addr in ip_map.items():
                if str(addr) == str(ip) and name.startswith("Enterprise"):
                    enterprise_host = name
                    break
            if enterprise_host is not None:
                break

        assert enterprise_host is not None, "Failed to observe Enterprise scan from CybORG B_line"
        return enterprise_host

    for user_host in ["User1", "User2", "User3", "User4"]:
        cyborg_enterprise = cyborg_enterprise_for_user(user_host)

        user_idx = HOST_IDS[user_host]
        try:
            user_pos = [i for i, x in enumerate(const.bline_user_hosts.tolist()) if x == user_idx][0]
        except IndexError as exc:
            raise AssertionError(f"{user_host} not found in bline_user_hosts") from exc

        jax_enterprise_idx = int(const.user_to_enterprise[user_pos])
        jax_enterprise = HOST_NAMES[jax_enterprise_idx]

        assert jax_enterprise == cyborg_enterprise, (
            f"{user_host}: JAX maps to {jax_enterprise}, CybORG uses {cyborg_enterprise}"
        )
