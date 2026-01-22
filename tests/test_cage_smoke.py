"""Smoke test for CAGE-JAX: verify basic state can JIT compile."""

import jax
import jax.numpy as jnp
from flax import struct
import chex


@struct.dataclass
class MinimalState:
    time: int
    host_compromised: chex.Array


def test_minimal_jit():
    """Verify basic CAGE-like state can JIT compile."""

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
    print("✓ Minimal CAGE state JIT-compiles")


if __name__ == "__main__":
    test_minimal_jit()
