import jax.numpy as jnp
from types import SimpleNamespace
from fft_tools import fft_sol_from_grid

def test_fft_sol_known_signal():
    # Create a synthetic sine wave solution
    t = jnp.linspace(0, 1, 1000)
    y = jnp.sin(2 * jnp.pi * 5 * t)
    sol = SimpleNamespace(ts=t, ys=jnp.stack([y, y], axis=-1))  # Two identical channels

    xf, yf, freqs, amps = fft_sol_from_grid(sol, i_array=[0])
    assert jnp.isclose(freqs[0], 5.0, atol=0.5), f"Expected ~5 Hz, got {freqs[0]}"