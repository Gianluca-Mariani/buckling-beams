# fft_tools.py

from jax import vmap
import jax.numpy as jnp
from jax.numpy.fft import fft, fftfreq
import matplotlib.pyplot as plt


def fft_single_component(state_values, ts, i):
    N = len(ts)
    T = ts[1] - ts[0]
    xf = fftfreq(N, T)[: N // 2]
    signal = state_values[:, i] - jnp.mean(state_values[:, i])
    yf = fft(signal)

    freq = jnp.abs(xf[jnp.argmax(jnp.abs(yf[: N // 2]))])
    amp = 2.0 / N * jnp.abs(yf[jnp.argmax(jnp.abs(yf[: N // 2]))])
    return yf, freq, amp


def fft_sol_from_grid(ys, ts, i_array):
    """
    Computes FFT in parallel across selected state components.
    Supports both single and multiple indices.
    """

    i_array = jnp.atleast_1d(jnp.array(i_array))

    # Make sure ys is 2D
    if ys.ndim == 1:
        ys = ys[:, None]  # Convert shape (N,) -> (N, 1)

    if len(i_array) == 0:
        raise ValueError("i_array must contain at least one index.")

    fft_vec = vmap(lambda i: fft_single_component(ys, ts, i))(i_array)

    # Stack the results along the first axis
    fft_results, dominant_frequencies, dominant_amplitudes = fft_vec

    return (
        fftfreq(len(ts), ts[1] - ts[0])[: len(ts) // 2],
        fft_results,
        dominant_frequencies,
        dominant_amplitudes,
    )
