#!/usr/bin/env python
"""Verify JAX installation for CAGE-JAX development."""

import jax
import jax.numpy as jnp
from flax import struct
import chex


def verify_jax():
    print(f"JAX version: {jax.__version__}")
    print(f"Devices: {jax.devices()}")
    print(f"Platform: {jax.devices()[0].platform}")

    # Test JIT compilation
    @jax.jit
    def add(x, y):
        return x + y

    result = add(jnp.array(1.0), jnp.array(2.0))
    assert result == 3.0, "JIT test failed"
    print("JIT compilation: OK")

    # Test vmap (parallel execution)
    @jax.jit
    @jax.vmap
    def batch_add(x):
        return x + 1.0

    batch = jnp.arange(1000)
    result = batch_add(batch)
    assert result.shape == (1000,), "vmap test failed"
    print("vmap parallelization: OK")

    # Test flax dataclass (used for state)
    @struct.dataclass
    class TestState:
        value: chex.Array
        done: bool

    state = TestState(value=jnp.zeros(10), done=False)
    new_state = state.replace(done=True)
    assert new_state.done == True, "flax dataclass test failed"
    print("flax dataclass: OK")

    print("\n✓ JAX environment ready for CAGE-JAX development")


if __name__ == "__main__":
    verify_jax()
