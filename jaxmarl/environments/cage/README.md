# CAGE Environment (JAX Implementation)

A JAX-accelerated implementation of the CAGE Challenge 2 environment, designed to match CybORG's behavior while enabling fast parallel simulation.

## Overview

This environment simulates a network security scenario where:
- **Red Agent**: Attacker starting with a foothold on User0, trying to reach and impact Op_Server0
- **Blue Agent**: Defender protecting all hosts via monitoring, removal, restoration, and decoys
- **Green Agent**: Background user activity (simulated implicitly)

## Architecture

```
jaxmarl/environments/cage/
├── cage_env.py          # Main environment (CageEnv)
├── state.py             # CageState, CageConst definitions
├── actions.py           # Action handling (Red & Blue)
├── observations.py      # Observation encoding
├── rewards.py           # Reward calculations
├── config.py            # ScenarioConfig and loaders
├── scripted_agents.py   # B_line, Meander agents
└── cyborg_loader.py     # Load config from CybORG YAML files
```

## Key Semantic Differences from CybORG

### IP Addresses vs Host Indices

**CybORG** uses IP addresses internally:
- `DiscoverRemoteSystems` reveals IP addresses in a subnet
- Red targets hosts by IP address
- `get_agent_state('Red')` returns info keyed by hostname only for hosts whose IPs Red has learned

**JAX** uses host indices directly:
- `red_discovered_hosts_jax[host_idx]` tracks if host is discovered (boolean)
- `red_scanned_hosts_jax[host_idx]` tracks if host is scanned (boolean)
- Actions target hosts by index

This abstraction simplifies the state representation while preserving equivalent game dynamics. The differential test suite verifies that rewards and action outcomes match CybORG exactly.

### Discovery and Scanning

CybORG's discovery model:
1. `DiscoverRemoteSystems` (DiscoverSubnet) - reveals IPs of hosts in target subnet
2. `DiscoverNetworkServices` (Scan) - reveals services/ports on a specific host
3. Exploit can succeed if Red knows the host (via either discovery or scan)

JAX matches this behavior:
- Scan requires routing (session in adjacent subnet), does NOT require prior DiscoverSubnet
- Exploit requires host to be known (`discovered OR scanned`)
- Cross-subnet scanning works from adjacent subnet sessions

### Detection Model

CybORG's Monitor action uses DensityScout to detect malicious processes:
- Only detects Exploit and PrivEsc activity (creates malware processes)
- Does NOT detect Scan activity (only creates network connections)
- 95% detection rate for detectable actions

JAX matches this with `ACTIVITY_EXPLOIT` threshold in Monitor/Analyse.

## Differential Testing

The `tests/comparison/` directory contains differential tests that:
1. Run both CybORG and JAX with identical action sequences
2. Compare rewards and state at each step
3. Verify 400+ test scenarios match

Run tests:
```bash
uv run pytest tests/comparison/ -v
```

Key test files:
- `test_cyborg_parity.py` - Comprehensive CybORG behavior tests
- `test_discovery_parity.py` - Discovery/scan semantics
- `test_smoke_parity.py` - B_line killchain trajectories

## Network Topology (Scenario 2)

```
User Subnet          Enterprise Subnet       Operational Subnet
┌─────────┐          ┌─────────────┐          ┌──────────────┐
│ User0*  │          │ Enterprise0 │          │ Op_Host0     │
│ User1   │◄────────►│ Enterprise1 │◄────────►│ Op_Host1     │
│ User2   │          │ Enterprise2 │          │ Op_Host2     │
│ User3   │          │ Defender    │          │ Op_Server0** │
│ User4   │          └─────────────┘          └──────────────┘
└─────────┘
* Red's initial foothold
** Impact target (value: 10.0)
```

- User ↔ Enterprise: directly connected
- Enterprise ↔ Operational: directly connected
- User cannot directly reach Operational

## Host Services

| Host | Services | Vulnerable To |
|------|----------|---------------|
| User0-4 | ssh | SSHBruteForce |
| User3 | ssh, haraka | SSHBruteForce, HarakaRCE |
| Enterprise0 | ssh, http | SSHBruteForce, HTTPRFI |
| Enterprise1 | ssh, smb | SSHBruteForce, EternalBlue |
| Enterprise2 | ssh, tomcat | SSHBruteForce, HTTPRFI |
| Op_Server0 | ssh, http | SSHBruteForce, HTTPRFI |

## Action Spaces

### Red Actions (143 total)
- Sleep (1)
- DiscoverRemoteSystems per subnet (3)
- DiscoverNetworkServices per host (13)
- Exploit per host × exploit type (13 × 8 = 104)
- PrivilegeEscalate per host (13)
- Impact per host (13) - only succeeds on Op_Server0

### Blue Actions (134 total)
- Sleep (1)
- Monitor (1)
- Analyse per host (13)
- Remove per host (13)
- Restore per host (13)
- Decoy per host × decoy type (13 × 8 = 104) - limited by existing services

## Rewards

- **Confidentiality**: Penalty when Red compromises hosts (value-based)
- **Availability**: -10.0 per step while Op_Server0 is impacted
- **Restore cost**: -1.0 per Restore action
- **Zero-sum**: `reward_red = -reward_blue`

## Usage

```python
from jaxmarl.environments.cage import CageEnv

env = CageEnv()
key = jax.random.PRNGKey(0)
obs, state = env.reset(key)

# Step with actions
actions = {'blue': blue_action, 'red': red_action}
obs, state, rewards, dones, info = env.step(key, state, actions)
```
