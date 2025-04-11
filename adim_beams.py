"""
This script defines classes to simulate the propagation of a soliton along buckling beam arrays with non-dimensional parameters
"""


import jax
import jax.numpy as jnp
import diffrax as dfx
from jax import vmap
from fft_tools import fft_sol_from_grid 
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time

# Core class: encapsulates the system and ODE solver
class AdimBeamSystem:
    def __init__(self, omega, r0, r1, rounding=2):
        self.omega = omega
        self.r0 = r0
        self.r1 = r1

    def phi(self, i):
        return jnp.where((i % 4 == 0) | (i % 4 == 3), 0.0, jnp.pi)

    def lagrangian(self, q, t, omega, r0, r1):
        n = len(q)
        time_factor = (-1)**jnp.arange(n) + r0 * jnp.sin(omega * t + self.phi(jnp.arange(n)))
        potential_quad = 0.5 * jnp.sum(time_factor * q**2)
        potential_quartic = 0.25 * jnp.sum(q**4)
        q_shifted = jnp.roll(q, shift=1)
        potential_coupling = 0.5 * r1 * jnp.sum((q - q_shifted)**2)
        return - (potential_quad + potential_quartic + potential_coupling)

    def ODEs(self, t, q, args):
        omega, r0, r1 = args
        # Compute the equations of motion using the Lagrangian
        return jax.grad(self.lagrangian, argnums=0)(q, t, omega, r0, r1)

    def solve(self, y0, omega, r0, r1, t_cycles=5, N_fact=2000):
        t0 = 0.0
        t1 = t_cycles * 2 * jnp.pi / omega
        ts = jnp.linspace(t0, t1, N_fact)
        args=(omega, r0, r1)
        saveat = dfx.SaveAt(ts=ts)
        solver = dfx.Tsit5()
        term = dfx.ODETerm(self.ODEs)
        sol = dfx.diffeqsolve(term, solver, t0, t1, ts[1] - ts[0], y0, saveat=saveat, args=args)
        return sol



# BeamAnalyzer class: encapsulates the analysis and plotting
class BeamAnalyzer:
    def __init__(self, sol, omega, r0, r1):
        self.sol = sol
        self.omega = omega
        self.r0 = r0
        self.r1 = r1


    def fft(self, i_array):
        xf, fft_results, freqs, amps = fft_sol_from_grid(self.sol, i_array)
        self.xf = xf
        self.fft_results = fft_results
        self.dominant_frequencies = freqs
        self.dominant_amplitudes = amps


    def time_series(self, i_array, limits=True):
        ts, ys = self.sol.ts, self.sol.ys
        for i in i_array:
            fig, ax = plt.subplots() 
            ax.plot(ts, ys[:, i], label="Numerical path")
            if limits:
                ax.axhline(y=jnp.sqrt(1 + self.r0), color='r', linestyle='--')
                ax.axhline(y=jnp.sqrt(1 - self.r0), color='r', linestyle='--')
            ax.set_xlabel("t")
            ax.set_ylabel(f"$x_{i}$")
            ax.set_title(fr"$x_{i}(t)$ ($\Omega={self.omega:.2f}$, $r_0={self.r0:.2f}$, $r_1={self.r1:.2f}$)")
            ax.legend()
            ax.grid(True)

    def phase_portrait(self, i1, i2, analytical=True):
        """Plots phase portraits for solutions"""
        fig, ax = plt.subplots() 
        state_values = self.sol.ys
        time_values = self.sol.ts
        an_sol = jnp.array([
            jnp.zeros(len(time_values)),
            jnp.sqrt(1+self.r0*jnp.sin(self.omega*time_values)),
            jnp.zeros(len(time_values)),
            jnp.sqrt(1-self.r0*jnp.sin(self.omega*time_values))
        ])
        ax.plot(an_sol[i1], an_sol[i2], color='r', linestyle='--', label=r"Analytical equilibrium path")
        ax.plot(state_values[:,i1], state_values[:,i2], label="Numerical path")
        
        ax.set_xlabel(f"$x_{i1}$")
        ax.set_ylabel(f"$x_{i2}$")
        ax.set_title(fr"Phase portrait on $(x_{i1},x_{i2})$ plane ($\Omega = {float(jnp.round(self.omega, 2)):.2f}$, $r_0 = {float(jnp.round(self.r0, 2)):.2f}$, $r_1 = {float(jnp.round(self.r1, 2)):.2f}$)")
        ax.legend(loc='lower left')
        ax.grid(True)

class beam_var:
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
        self.omegas = jnp.kron(jnp.kron(omegas[:, None, None], jnp.ones(self.lengths[1])[None, :, None]), jnp.ones(self.lengths[2])[None, None, :])  # 3d array of compression/decompression frequency
        self.r0s = jnp.kron(jnp.kron(jnp.ones(self.lengths[0])[:, None, None], r0s[None, :, None]), jnp.ones(self.lengths[2])[None, None, :])        # 3d array of relative comression amplitude (if r0<0 no phase transition for single beam)        
        self.r1s = jnp.kron(jnp.kron(jnp.ones(self.lengths[0])[:, None, None], jnp.ones(self.lengths[1])[None, :, None]), r1s[None, None, :])        # 3d array of coupling vs local stiffness ratio
        self.beams_param = [[[adim_beams(self.omegas[i, j, k], self.r0s[i, j, k], self.r1s[i, j, k]) for k in range(self.lengths[2])]\
               for j in range(self.lengths[1])] for i in range(self.lengths[0])]  # 3d array of adim_beams with every combination of parameters

    def solve_objects(self, y0):
        for i in range(self.lengths[0]):
            for j in range(self.lengths[1]):
                for k in range(self.lengths[2]):
                    self.beams_param[i][j][k].solve_lagrangian_system(y0)

    def solve_ffts(self, i_array, plot_bool=False, N_max=100):
        for i in range(self.lengths[0]):
            for j in range(self.lengths[1]):
                for k in range(self.lengths[2]):
                    self.beams_param[i][j][k].fft_sol(i_array, plot_bool=plot_bool, N_max=N_max) 
    
    def time_plots(self, i_array, limits=False):
        for i in range(self.lengths[0]):
            for j in range(self.lengths[1]):
                for k in range(self.lengths[2]):
                    self.beams_param[i][j][k].time_series_plot(i_array=i_array, limits=limits)

    def phase_plots(self, i1, i2, analytical = False):
        for i in range(self.lengths[0]):
            for j in range(self.lengths[1]):
                for k in range(self.lengths[2]):
                     self.beams_param[i][j][k].phase_portrait_plot(i1=i1, i2=i2, analytical = analytical)

    def vary_param(self, function_to_run, y_arr, p1, p2=None):
        """"
        Plots a quantity (defined by function_to_run) as function of chosen parameter p1.
        If p2 is not None, multiple plots are produced on the same figure, each with a different value of p2.
        The others parameters are averaged over.
        p1, p2 can be "omega", "r0", "r1"
        """

        indices_dict = {"omega" : 0, "r0" : 1, "r1" : 2}
        try:
            q1 = indices_dict[p1]
        except:
            raise Exception("Paramater 1 type not supported")
        
        for y in y_arr:
            # Convert calculated values to jax arrays, which allow better slicing
            t_values = jnp.array([[[beam.sol.ts for beam in row] for row in layer] for layer in self.beams_param])
            y_values = jnp.array([[[beam.sol.ys[y] for beam in row] for row in layer] for layer in self.beams_param])
            freq_values = jnp.array([[[beam.dominant_frequencies[y] for beam in row] for row in layer] for layer in self.beams_param])
            ampl_values = jnp.array([[[beam.dominant_amplitudes[y] for beam in row] for row in layer] for layer in self.beams_param])
            
            if p2 == None:
                t_avg = jnp.mean(t_values, axis=tuple(j for j in range(3) if j != q1))
                y_avg = jnp.mean(y_values, axis=tuple(j for j in range(3) if j != q1))
                freq_avg = jnp.mean(freq_values, axis=tuple(j for j in range(3) if j != q1))
                ampl_avg = jnp.mean(ampl_values, axis=tuple(j for j in range(3) if j != q1))
                function_to_run(q1, None, t_avg, y_avg, freq_avg, ampl_avg, [self.omegas[:, 0, 0], self.r0s[0, :, 0], self.r1s[0, 0, :]], y)

            else:
                try:
                    q2 = indices_dict[p2]
                except:
                    raise Exception("Paramater 2 type not supported") 
                
                all_axes = [0, 1, 2]
                fixed_axes = [q1, q2]
                axes_to_average = [axis for axis in all_axes if axis not in fixed_axes][0]

                t_avg = jnp.mean(t_values, axis=axes_to_average)
                y_avg = jnp.mean(y_values, axis=axes_to_average)
                freq_avg = jnp.mean(freq_values, axis=axes_to_average)
                ampl_avg = jnp.mean(ampl_values, axis=axes_to_average)
                function_to_run(q1, q2, t_avg, y_avg, freq_avg, ampl_avg, [self.omegas[:, 0, 0], self.r0s[0, :, 0], self.r1s[0, 0, :]], y)


#/(jnp.sqrt(1+params[q2][i])-jnp.sqrt(1-params[q2][i]))*2
#/A_{{eq}}

def A_vs_Omega(q1, q2, t_avg, y_avg, freq_avg, ampl_avg, params, y_curr):
    fig, ax = plt.subplots() 
    if q2 == None:
            ax.plot(params[q1], ampl_avg, label="Numerical values")
    else:
        label1 = {0: r"$\Omega = $", 1: r"$r_0 = $", 2: r"$r_1 = $"}[q2] 
        colors = cm.magma(jnp.linspace(0.1, 0.9, len(params[q2])))
        for i in range(len(params[q2])):
            if q1 < q2:
                ax.plot(params[q1], ampl_avg[:,i], label= label1 + f"${float(jnp.round(params[q2][i], 2)):.2f}$", color=colors[i], linewidth=1.0, marker='o', markersize=3)
            else:
                ax.plot(params[q1], ampl_avg[i,:], label= label1 + f"${float(jnp.round(params[q2][i], 2)):.2f}$", color=colors[i], linewidth=1.0, marker='o', markersize=3)

    if q1 == 0:
        ax.set_xlabel(r"$\Omega$")
    elif q1 == 1:
        ax.set_xlabel(r"$r_0$")
    elif q1 == 2:
        ax.set_xlabel(r"$r_1$")
    ax.set_ylabel(fr"$A_{{{y_curr}}}$")
    ax.legend(loc='upper right')
    ax.grid(True)
    ax.set_xscale('log')
