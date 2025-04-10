# fft_tools.py

import jax.numpy as jnp
from jax.numpy.fft import fft, fftfreq
import matplotlib.pyplot as plt

def fft_sol_from_grid(sol, i_array):
    """
    Computes FFT of selected solution components from a solved system using JAX, can be run in parallel.
    
    Parameters:
    - sol: An object with `ts` and `ys` attributes (e.g. from diffeqsolve).
    - i_array: List of indices to apply FFT to (e.g., [1, 3]).
    
    Returns:
    - xf, fft_results, dominant_frequencies, dominant_amplitudes (as jnp arrays)
    """

    time_values = sol.ts
    state_values = sol.ys
    N = len(time_values)
    T = time_values[1] - time_values[0]
    xf = fftfreq(N, T)[:N // 2]
    fft_results = []

    for i in i_array:
        yf = fft(state_values[:, i] - jnp.mean(state_values[:, i]))
        fft_results.append(yf)

    fft_results = jnp.array(fft_results)

    dominant_frequencies = jnp.array([
        jnp.abs(xf[jnp.argmax(jnp.abs(yf[:N // 2]))]) for yf in fft_results
    ])

    dominant_amplitudes = jnp.array([
        2.0 / N * jnp.abs(yf[jnp.argmax(jnp.abs(yf[:N // 2]))]) for yf in fft_results
    ])
    
    return xf, fft_results, dominant_frequencies, dominant_amplitudes