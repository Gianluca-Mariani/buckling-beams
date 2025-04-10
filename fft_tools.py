# fft_tools.py

import jax.numpy as jnp
from jax.numpy.fft import fft, fftfreq
import matplotlib.pyplot as plt

def fft_sol_from_grid(sol, i_array, rounding=2, plot_bool=True, N_max=100, omega=None, r0=None, r1=None):
    """
    Computes FFT of selected solution components from a solved system.
    
    Parameters:
    - sol: An object with `ts` and `ys` attributes (e.g. from diffeqsolve).
    - i_array: List of indices to apply FFT to (e.g., [1, 3]).
    - rounding: Decimal rounding for titles/printouts.
    - plot_bool: Whether to show FFT plots.
    - N_max: Max number of frequency components to plot.
    - omega, r0, r1: Optional parameters to display in titles.
    
    Returns:
    - dominant_frequencies, dominant_amplitudes (as jnp arrays)
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

    if plot_bool:
        for i in range(len(i_array)):
            fig, ax = plt.subplots()
            ax.plot(
                2 * jnp.pi * xf[:N_max],
                (2.0 / N * jnp.abs(fft_results[i][:N // 2]))[:N_max],
                label="FFT Magnitude"
            )

            title = fr"FFT $x_{{{i_array[i]}}}$"
            if omega is not None and r0 is not None and r1 is not None:
                title += fr" ($\Omega = {float(jnp.round(omega, rounding)):.2f}$, $r_0 = {float(jnp.round(r0, rounding)):.2f}$, $r_1 = {float(jnp.round(r1, rounding)):.2f}$)"
                print(f"Dominant Frequency x_{i_array[i]}: {float(dominant_frequencies[i]):.2f}")
                print(f"Dominant Amplitude x_{i_array[i]}: {dominant_amplitudes[i]}")
            
            ax.set_title(title)
            ax.set_xlabel(r"Frequency $\omega$")
            ax.set_ylabel("Amplitude")
            ax.grid(True)
    
    return dominant_frequencies, dominant_amplitudes