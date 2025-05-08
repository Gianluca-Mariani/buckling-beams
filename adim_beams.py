"""
This script defines classes to simulate the propagation of a soliton along buckling beam arrays with non-dimensional parameters
"""

import jax
import jax.numpy as jnp
import diffrax as dfx
from fft_tools import fft_sol_from_grid
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from itertools import product
import time

# Base functions for simulating system dynamics


def phi(i):
    return jnp.where((i % 4 == 0) | (i % 4 == 3), 0.0, jnp.pi)


def lagrangian(q, t, omega, r0, r1):
    n = q.shape[0]
    time_factor = (-1) ** jnp.arange(n) + r0 * jnp.sin(omega * t + phi(jnp.arange(n)))
    potential_quad = 0.5 * jnp.sum(time_factor * q**2)
    potential_quartic = 0.25 * jnp.sum(q**4)
    q_shifted = jnp.roll(q, shift=1)
    potential_coupling = 0.5 * r1 * jnp.sum((q - q_shifted) ** 2)
    return -(potential_quad + potential_quartic + potential_coupling)


def ODEs(t, q, omega, r0, r1):
    return jax.grad(lagrangian, argnums=0)(q, t, omega, r0, r1)


def solve_system(y0, omega, r0, r1, t_cycles=5, N_fact=2000):
    t0 = 0.0
    t1 = t_cycles * 2 * jnp.pi / omega
    ts = jnp.linspace(t0, t1, N_fact)
    saveat = dfx.SaveAt(ts=ts)
    solver = dfx.Tsit5()
    term = dfx.ODETerm(lambda t, y, args: ODEs(t, y, *args))
    sol = dfx.diffeqsolve(
        term, solver, t0, t1, ts[1] - ts[0], y0, saveat=saveat, args=(omega, r0, r1)
    )
    return sol


# BeamAnalyzer class: encapsulates the analysis and plotting
class BeamAnalyzer:
    def __init__(self, sol_t, sol_y, omega, r0, r1):
        self.sol_t = sol_t
        self.sol_y = sol_y
        self.omega = omega
        self.r0 = r0
        self.r1 = r1

    def fft(self, i_array):
        xf, fft_results, freqs, amps = fft_sol_from_grid(
            self.sol_y, self.sol_t, i_array
        )
        self.xf = xf
        self.fft_results = fft_results
        self.dominant_frequencies = freqs
        self.dominant_amplitudes = amps

    def plot_fft(self, i_array, N_max=100):
        for i in i_array:
            fig, ax = plt.subplots()
            ax.plot(self.xf[:N_max], jnp.abs(self.fft_results[i][:N_max]), label="FFT")
            ax.axvline(
                x=self.dominant_frequencies[i],
                color="r",
                linestyle="--",
                label="Dominant frequency",
            )
            ax.set_xlabel("Frequency")
            ax.set_ylabel(f"FFT of $x_{i}$")
            ax.set_title(
                rf"FFT of $x_{i}$ ($\Omega={self.omega:.2f}$, $r_0={self.r0:.2f}$, $r_1={self.r1:.2f}$)"
            )
            ax.legend()
            ax.grid(True)
            print(
                f"Dominant frequency for x_{i}: {2 * jnp.pi * self.dominant_frequencies[i]:.2f}"
            )
            print(f"Dominant amplitude for x_{i}: {self.dominant_amplitudes[i]:.2f}")
        pass

    def time_series(self, i_array, limits=True):
        for i in i_array:
            fig, ax = plt.subplots()
            ax.plot(self.sol_t, self.sol_y[:, i], label="Numerical path")
            if limits:
                ax.axhline(y=jnp.sqrt(1 + self.r0), color="r", linestyle="--")
                ax.axhline(y=jnp.sqrt(1 - self.r0), color="r", linestyle="--")
            ax.set_xlabel("t")
            ax.set_ylabel(f"$x_{i}$")
            ax.set_title(
                rf"$x_{i}(t)$ ($\Omega={self.omega:.2f}$, $r_0={self.r0:.2f}$, $r_1={self.r1:.2f}$)"
            )
            ax.legend()
            ax.grid(True)

    def phase_portrait(self, i1, i2, analytical=True):
        """Plots phase portraits for solutions"""
        fig, ax = plt.subplots()
        an_sol = jnp.array(
            [
                jnp.zeros(len(self.sol_t)),
                jnp.sqrt(1 + self.r0 * jnp.sin(self.omega * self.sol_t)),
                jnp.zeros(len(self.sol_t)),
                jnp.sqrt(1 - self.r0 * jnp.sin(self.omega * self.sol_t)),
            ]
        )
        ax.plot(
            an_sol[i1],
            an_sol[i2],
            color="r",
            linestyle="--",
            label=r"Analytical equilibrium path",
        )
        ax.plot(self.sol_y[:, i1], self.sol_y[:, i2], label="Numerical path")

        ax.set_xlabel(f"$x_{i1}$")
        ax.set_ylabel(f"$x_{i2}$")
        ax.set_title(
            rf"Phase portrait on $(x_{i1},x_{i2})$ plane ($\Omega = {float(jnp.round(self.omega, 2)):.2f}$, $r_0 = {float(jnp.round(self.r0, 2)):.2f}$, $r_1 = {float(jnp.round(self.r1, 2)):.2f}$)"
        )
        ax.legend(loc="lower left")
        ax.grid(True)


class AdimBeamSystemArray:
    """
    Creates multiple instances of adim_beams, with different parameters
    Requires diffrax, jax, jax.numpy and matplotlib.pyplot
    """

    def __init__(self, omegas, r0s, r1s):
        """
        Initialize the physical parameters array
        Kronecker products used to sweep every combination of the input parameters
        """
        self.lengths = jnp.array([len(omegas), len(r0s), len(r1s)])
        self.omegas = jnp.kron(
            jnp.kron(omegas[:, None, None], jnp.ones(self.lengths[1])[None, :, None]),
            jnp.ones(self.lengths[2])[None, None, :],
        )  # 3d array of compression/decompression frequency
        self.r0s = jnp.kron(
            jnp.kron(jnp.ones(self.lengths[0])[:, None, None], r0s[None, :, None]),
            jnp.ones(self.lengths[2])[None, None, :],
        )  # 3d array of relative comression amplitude (if r0<1 no phase transition for single beam)
        self.r1s = jnp.kron(
            jnp.kron(
                jnp.ones(self.lengths[0])[:, None, None],
                jnp.ones(self.lengths[1])[None, :, None],
            ),
            r1s[None, None, :],
        )  # 3d array of coupling vs local stiffness ratios

    def solve(self, y0, t_cycles=5, N_fact=2000):
        """
        Solves the system of equations for all combinations of parameters
        y0: initial conditions
        """
        # Create a list of parameter combinations
        # Flatten combinations into a list
        params = list(
            product(
                range(self.lengths[0]), range(self.lengths[1]), range(self.lengths[2])
            )
        )

        # You can vmap or parallelize over these
        @jax.vmap
        def solve_wrapper(idx):
            i, j, k = idx
            return solve_system(
                y0,
                self.omegas[i, j, k],
                self.r0s[i, j, k],
                self.r1s[i, j, k],
                t_cycles=t_cycles,
                N_fact=N_fact,
            )

        # Unpack the results
        results = solve_wrapper(jnp.array(params))
        # Reshape ts and ys first dimension into 3D arrays
        self.ts = jnp.reshape(
            results.ts, tuple(self.lengths) + (results.ts.shape[-1],)
        )  # Reshape ts to a 3D array of shape (len(omegas), len(r0s), len(r1s))
        self.ys = jnp.reshape(
            results.ys,
            tuple(self.lengths) + (results.ys.shape[-2], results.ys.shape[-1]),
        )  # Reshape ts to a 3D array of shape (len(omegas), len(r0s), len(r1s))

    def solve_ffts(self, i_array, plot_bool=False, N_max=100):
        """
        Computes the FFT for all combinations of parameters
        """
        # Create a list of parameter combinations
        # Flatten combinations into a list
        params = list(
            product(
                range(self.lengths[0]), range(self.lengths[1]), range(self.lengths[2])
            )
        )

        # You can vmap or parallelize over these
        @jax.vmap
        def solve_wrapper(idx):
            i, j, k = idx
            return fft_sol_from_grid(self.ys[i, j, k], self.ts[i, j, k], i_array)

        # Unpack the results
        xf, ff, df, da = solve_wrapper(jnp.array(params))

        # Reshape ts and ys first dimension into 3D arrays
        self.xf = jnp.reshape(xf, tuple(self.lengths) + (xf.shape[-1],))
        self.fft_results = jnp.reshape(
            ff, tuple(self.lengths) + (ff.shape[-2], ff.shape[-1])
        )
        self.dominant_frequencies = jnp.reshape(
            df, tuple(self.lengths) + (df.shape[-1],)
        )
        self.dominant_amplitudes = jnp.reshape(
            da, tuple(self.lengths) + (da.shape[-1],)
        )

        if plot_bool:
            for i in range(self.lengths[0]):
                for j in range(self.lengths[1]):
                    for k in range(self.lengths[2]):
                        system = BeamAnalyzer(
                            self.ts[i, j, k],
                            self.ys[i, j, k],
                            self.omegas[i, j, k],
                            self.r0s[i, j, k],
                            self.r1s[i, j, k],
                        )
                        system.xf = self.xf[i, j, k]
                        system.fft_results = self.fft_results[i, j, k]
                        system.dominant_frequencies = self.dominant_frequencies[i, j, k]
                        system.dominant_amplitudes = self.dominant_amplitudes[i, j, k]
                        system.plot_fft(i_array, N_max=N_max)

    def time_plots(self, i_array, limits=False):
        for i in range(self.lengths[0]):
            for j in range(self.lengths[1]):
                for k in range(self.lengths[2]):
                    system = BeamAnalyzer(
                        self.ts[i, j, k],
                        self.ys[i, j, k],
                        self.omegas[i, j, k],
                        self.r0s[i, j, k],
                        self.r1s[i, j, k],
                    )
                    system.time_series(i_array=i_array, limits=limits)

    def phase_plots(self, i1, i2, analytical=False):
        for i in range(self.lengths[0]):
            for j in range(self.lengths[1]):
                for k in range(self.lengths[2]):
                    system = BeamAnalyzer(
                        self.ts[i, j, k],
                        self.ys[i, j, k],
                        self.omegas[i, j, k],
                        self.r0s[i, j, k],
                        self.r1s[i, j, k],
                    )
                    system.phase_portrait(i1=i1, i2=i2, analytical=analytical)

    def phase_plots_old(self, i1, i2, analytical=False):
        for i in range(self.lengths[0]):
            for j in range(self.lengths[1]):
                for k in range(self.lengths[2]):
                    self.beams_param[i][j][k].phase_portrait_plot(
                        i1=i1, i2=i2, analytical=analytical
                    )

    def vary_param(self, function_to_run, y_arr, p1, p2=None):
        """ "
        Plots a quantity (defined by function_to_run) as function of chosen parameter p1.
        If p2 is not None, multiple plots are produced on the same figure, each with a different value of p2.
        The others parameters are averaged over.
        p1, p2 can be "omega", "r0", "r1"
        """

        indices_dict = {"omega": 0, "r0": 1, "r1": 2}
        try:
            q1 = indices_dict[p1]
        except:
            raise Exception("Paramater 1 type not supported")

        if p2 == None:
            t_avg = jnp.mean(self.ts, axis=tuple(j for j in range(3) if j != q1))
            y_avg = jnp.mean(self.ys, axis=tuple(j for j in range(3) if j != q1))
            freq_avg = jnp.mean(
                self.dominant_frequencies, axis=tuple(j for j in range(3) if j != q1)
            )
            ampl_avg = jnp.mean(
                self.dominant_amplitudes, axis=tuple(j for j in range(3) if j != q1)
            )

        else:
            try:
                q2 = indices_dict[p2]
            except:
                raise Exception("Paramater 2 type not supported")

            all_axes = [0, 1, 2]
            fixed_axes = [q1, q2]
            axes_to_average = [axis for axis in all_axes if axis not in fixed_axes][0]

            t_avg = jnp.mean(self.ts, axis=axes_to_average)
            y_avg = jnp.mean(self.ys, axis=axes_to_average)
            freq_avg = jnp.mean(self.dominant_frequencies, axis=axes_to_average)
            ampl_avg = jnp.mean(self.dominant_amplitudes, axis=axes_to_average)

        for y in y_arr:
            if p2 == None:
                function_to_run(
                    q1,
                    None,
                    t_avg,
                    y_avg[:, :, y],
                    freq_avg[:, y_arr.index(y)],
                    ampl_avg[:, y_arr.index(y)],
                    [self.omegas[:, 0, 0], self.r0s[0, :, 0], self.r1s[0, 0, :]],
                    y,
                )
            else:
                function_to_run(
                    q1,
                    q2,
                    t_avg,
                    y_avg[:, :, :, y],
                    freq_avg[:, :, y_arr.index(y)],
                    ampl_avg[:, :, y_arr.index(y)],
                    [self.omegas[:, 0, 0], self.r0s[0, :, 0], self.r1s[0, 0, :]],
                    y,
                )


def A_vs_Omega(q1, q2, t_avg, y_avg, freq_avg, ampl_avg, params, y_curr):
    fig, ax = plt.subplots()
    if q2 == None:
        ax.plot(
            params[q1],
            ampl_avg,
            label="Numerical values",
            linewidth=1.0,
            marker="o",
            markersize=3,
        )
    else:
        label1 = {0: r"$\Omega = $", 1: r"$r_0 = $", 2: r"$r_1 = $"}[q2]
        colors = cm.magma(jnp.linspace(0.1, 0.9, len(params[q2])))
        for i in range(len(params[q2])):
            if q1 < q2:
                ax.plot(
                    params[q1],
                    ampl_avg[:, i],
                    label=label1 + f"${float(jnp.round(params[q2][i], 2)):.2f}$",
                    color=colors[i],
                    linewidth=1.0,
                    marker="o",
                    markersize=3,
                )
            else:
                ax.plot(
                    params[q1],
                    ampl_avg[i, :],
                    label=label1 + f"${float(jnp.round(params[q2][i], 2)):.2f}$",
                    color=colors[i],
                    linewidth=1.0,
                    marker="o",
                    markersize=3,
                )

    if q1 == 0:
        ax.set_xlabel(r"$\Omega$")
    elif q1 == 1:
        ax.set_xlabel(r"$r_0$")
    elif q1 == 2:
        ax.set_xlabel(r"$r_1$")
    ax.set_ylabel(rf"$A_{{{y_curr}}}/A_{{eq}}$")
    ax.legend(loc="upper right")
    ax.grid(True)
    ax.set_xscale("log")
