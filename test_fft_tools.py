import jax.numpy as jnp
from types import SimpleNamespace
from fft_tools import fft_sol_from_grid


def test_fft_sol_known_signal_single():
    t = jnp.linspace(0, 1, 1000)
    y = jnp.sin(2 * jnp.pi * 5 * t)  # 5 Hz sine wave
    sol = SimpleNamespace(ts=t, ys=y)

    xf, yf, freqs, amps = fft_sol_from_grid(sol, i_array=[0])
    assert freqs.shape == (1, ), f"Expected one frequency, got shape {freqs.shape}"
    assert jnp.isclose(freqs, 5.0, atol=0.5), f"Expected ~5 Hz, got {freqs}"


def test_fft_sol_known_signal_multiple():
    t = jnp.linspace(0, 1, 1000)
    y1 = jnp.sin(2 * jnp.pi * 5 * t)   # 5 Hz
    y2 = jnp.sin(2 * jnp.pi * 10 * t)  # 10 Hz
    ys = jnp.stack([y1, y2], axis=-1)
    sol = SimpleNamespace(ts=t, ys=ys)

    xf, yf, freqs, amps = fft_sol_from_grid(sol, i_array=[0, 1])
    assert freqs.shape == (2,), f"Expected 2 frequencies, got {freqs.shape}"
    assert jnp.isclose(freqs[0], 5.0, atol=0.5), f"Expected ~5 Hz, got {freqs[0]}"
    assert jnp.isclose(freqs[1], 10.0, atol=0.5), f"Expected ~10 Hz, got {freqs[1]}"