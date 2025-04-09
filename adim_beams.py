"""
This script defines classes to simulate the propagation of a soliton along buckling beam arrays with non-dimensional parameters
"""

# Import packages
import diffrax as dfx
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.fft import fft, fftfreq
import time

class adim_beams:
    """
    Propagates a soliton through a beam array, given the adimensional parameters and initial condition
    Requires diffrax, jax, jax.numpy and matplotlib.pyplot
    """
    def __init__(self, omega, r0, r1):
        """Initialize the physical parameters"""
        self.omega = omega    # Compression/decompression frequency
        self.r0 = r0          # Relative comression amplitude (if r0<0 no phase transition for single beam)
        self.r1 = r1          # Coupling vs local stiffness ratio
        self.rounding = 2     # Rounding decimal places

    def phi(self, i):
        """Phase function: 0 if i%4 == 0 or i%4 == 3, otherwise π."""
        return jnp.where((i % 4 == 0) | (i % 4 == 3), 0.0, jnp.pi)

    def lagrangian(self, q, t):
        """Lagrangian for an n-degree-of-freedom system with cyclic boundaries and no mass"""

        # Time-dependent quadratic potential
        time_factor = (-1) ** jnp.arange(len(q)) + self.r0 * jnp.sin(self.omega * t + self.phi(jnp.arange(len(q))))
        potential_quad = 0.5 * jnp.sum(time_factor * q**2)

        # Quartic potential
        potential_quartic = 0.25 * jnp.sum(q**4)

        # Coupling term (cyclic boundary condition)
        q_shifted = jnp.roll(q, shift=1)  # Shift right (x[i-1] wraps around)
        potential_coupling = 0.5 * self.r1 * jnp.sum((q - q_shifted) ** 2)

        # Total potential energy
        potential = potential_quad + potential_quartic + potential_coupling

        return - potential  # L = - V
    

    def ODEs(self, t, q, args):
        """Compute the equations of motion"""

        return jax.grad(self.lagrangian, argnums=0)(q, t)  
    

    def solve_lagrangian_system(self, y0):
        """Solves the system with the Euler-Lagrange equation and dissipation."""

        solver = dfx.Tsit5()
        t5 = 10*jnp.pi/self.omega
        N_fact = 2000
        t0, t1 = 0.0, t5
        ts = jnp.linspace(t0, t1, N_fact)
        saveat = dfx.SaveAt(ts=ts)

        self.sol = dfx.diffeqsolve(
            dfx.ODETerm(self.ODEs), solver, t0, t1, ts[1]-ts[0], y0, saveat=saveat, args=None
        )

    def fft_sol(self, i_array, plot_bool=True, N_max=100):
        """Computes fft of the selected solution components"""
        time_values = self.sol.ts
        state_values = self.sol.ys
        N = len(time_values)
        T = time_values[1] - time_values[0]
        xf = fftfreq(N, T)[:N // 2]  # Frequency bins
        fft_results = []  # List to store FFT results

        for i in i_array:
            # Compute the FFT
            yf = fft(state_values[:,i] - jnp.mean(state_values[:,i]))
            fft_results.append(yf)  # Append FFT result for each component

        fft_results = jnp.array(fft_results)

        # Get dominant frequency and amplitude for each component
        self.dominant_frequencies = jnp.array([
                                    jnp.abs(xf[jnp.argmax(jnp.abs(yf[:N // 2]))]) for yf in fft_results
                                    ])
    
        self.dominant_amplitudes = jnp.array([
                                    2.0 / N * jnp.abs(yf[jnp.argmax(jnp.abs(yf[:N // 2]))]) for yf in fft_results
                                    ])
        
        # Plot the FFT result
        if plot_bool:
            for i in range(len(i_array)):
                fig, ax = plt.subplots()
                ax.plot(2*jnp.pi*xf[:N_max], (2.0 / N * jnp.abs((fft_results[i])[:N // 2]))[:N_max], label="FFT Magnitude")
                ax.set_title(fr"FFT $x_{i_array[i]}$ ($\Omega = {float(jnp.round(self.omega, self.rounding)):.2f}$, $r_0 = {float(jnp.round(self.r0, self.rounding)):.2f}$, $r_1 = {float(jnp.round(self.r1, self.rounding)):.2f}$)")
                ax.set_xlabel(r"Frequency $\omega$")
                ax.set_ylabel("Amplitude")
                print(f"Dominant Frequency x_{i_array[i]} (omega = {float(jnp.round(self.omega, self.rounding)):.2f}, r_0 = {float(jnp.round(self.r0, self.rounding)):.2f}, r_1 = {float(jnp.round(self.r1, self.rounding)):.2f}): {float(self.dominant_frequencies[i]):.2f}")
                print(f"Dominant Amplitude x_{i_array[i]} (omega = {float(jnp.round(self.omega, self.rounding)):.2f}, r_0 = {float(jnp.round(self.r0, self.rounding)):.2f}, r_1 = {float(jnp.round(self.r1, self.rounding)):.2f}): {self.dominant_amplitudes[i]}")
                ax.grid(True)


    
    def time_series_plot(self, i_array, limits=True):
        """Plots time series for solutions"""
        time_values = self.sol.ts
        state_values = self.sol.ys
        N = len(time_values)
        T = time_values[1] - time_values[0]
        for i in i_array:
            fig, ax = plt.subplots() 
            ax.plot(time_values, state_values[:,i], label=f"Numerical path")
            if limits:
                ax.axhline(y=jnp.sqrt(1+self.r0), color='r', linestyle='--', label=r"$\sqrt{1 \pm r_0}$")
                ax.axhline(y=jnp.sqrt(1-self.r0), color='r', linestyle='--')

            ax.set_xlabel("t")
            ax.set_ylabel(f"$x_{i}$")
            ax.set_title(fr"$x_{i} (t)$ ($\Omega = {float(jnp.round(self.omega, self.rounding)):.2f}$, $r_0 = {float(jnp.round(self.r0, self.rounding)):.2f}$, $r_1 = {float(jnp.round(self.r1, self.rounding)):.2f}$)")
            ax.legend(loc='upper left')
            ax.grid(True)
    
    
    def phase_portrait_plot(self, i1, i2, analytical=True):
        """Plots phase portraits for solutions"""
        fig, ax = plt.subplots() 
        state_values = self.sol.ys
        if analytical:
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
        ax.set_title(fr"Phase portrait on $(x_{i1},x_{i2})$ plane ($\Omega = {float(jnp.round(self.omega, self.rounding)):.2f}$, $r_0 = {float(jnp.round(self.r0, self.rounding)):.2f}$, $r_1 = {float(jnp.round(self.r1, self.rounding)):.2f}$)")
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


n=1
y00 = jnp.array([0])
y01 = jnp.array([1])
y02 = jnp.array([0])
y03 = jnp.array([1])
y0b = jnp.concatenate([y00, y01, y02, y03])
y0 = jnp.tile(y0b, n)


r0s = jnp.linspace(0.2, 0.8, 3)
r1s = jnp.array([0])
omegas = jnp.geomspace(1, 1, 1)


start_time = time.time()

uncoupled_sweep = beam_var(omegas=omegas, r0s=r0s, r1s=r1s)
uncoupled_sweep.solve_objects(y0)
uncoupled_sweep.solve_ffts([0, 1, 2, 3], plot_bool=False)
uncoupled_sweep.time_plots([1, 3], limits=True)
uncoupled_sweep.phase_plots(1, 3, analytical=True)
#uncoupled_sweep.vary_param(A_vs_Omega, [0, 1, 2, 3], "omega", "r0")


end_time = time.time()

execution_time = end_time - start_time
print(f"Execution time: {execution_time:.2f} seconds")


plt.show()