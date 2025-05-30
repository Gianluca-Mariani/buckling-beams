import jax.numpy as jnp
import numpy as np
from adim_beams import BeamAnalyzer, AdimBeamSystemArray
from adim_beams import solve_system, A_vs_Omega
import matplotlib.pyplot as plt
import time
from julia import Julia
from julia import Main

Main.include("find_equilibria.jl")
MyJulia = Main.FindEquilibria

n = 4
omega = 1.0
T = 2*np.pi / omega
times = np.linspace(0.0, T, 100)
r0 = 0.5
r1 = 0.2
aligned, real_result, stable_real_result = MyJulia.get_solutions_flags(n, times, omega, r0, r1)
aligned_np = np.swapaxes(np.array(aligned), 0, 1)
real_result_np = np.swapaxes(np.array(real_result), 0, 1)
stable_real_result_np = np.swapaxes(np.array(stable_real_result), 0, 1)
unstable_real_result_np = real_result_np & ~stable_real_result_np
print(stable_real_result_np)

fig0, ax0 = plt.subplots()
fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
fig3, ax3 = plt.subplots()
fig4, ax4 = plt.subplots()

for (i, sol) in enumerate(aligned_np):
    masked_sol = np.real(sol[stable_real_result_np[i]])
    masked_times = times[stable_real_result_np[i]]

    if len(masked_sol) > 0:
        ax0.plot(masked_times, masked_sol[:,0], marker='o', linestyle='None', markersize=5)
        ax1.plot(masked_times, masked_sol[:,1], marker='o', linestyle='None', markersize=5)
        ax2.plot(masked_times, masked_sol[:,2], marker='o', linestyle='None', markersize=5)
        ax3.plot(masked_times, masked_sol[:,3], marker='o', linestyle='None', markersize=5)
        ax4.plot(masked_sol[:,0], masked_sol[:,1], linestyle='-', linewidth=3)

    unstable_sol = np.real(sol[unstable_real_result_np[i]])
    unstable_times = times[unstable_real_result_np[i]]

    if len(unstable_sol) > 0:
        ax0.plot(unstable_times, unstable_sol[:,0], marker='x', linestyle='None', markersize=1)
        ax1.plot(unstable_times, unstable_sol[:,1], marker='x', linestyle='None', markersize=1)
        ax2.plot(unstable_times, unstable_sol[:,2], marker='x', linestyle='None', markersize=1)
        ax3.plot(unstable_times, unstable_sol[:,3], marker='x', linestyle='None', markersize=1)
        ax4.plot(unstable_sol[:,0], unstable_sol[:,1], linestyle=':', linewidth=1)

plt.show()

'''
# This part tests a single combination of parameters
n=1
y00 = jnp.array([0])
y01 = jnp.array([1])
y02 = jnp.array([0])
y03 = jnp.array([1])
y0b = jnp.concatenate([y00, y01, y02, y03])
y0 = jnp.tile(y0b, n)
 
test_sol = solve_system(y0, omega=1.0, r0=0.5, r1=0, t_cycles=5, N_fact=2000)
test_analyzer = BeamAnalyzer(test_sol.ts, test_sol.ys, omega=1.0, r0=0.5, r1=0)
test_analyzer.fft([1, 3])
test_analyzer.plot_fft([1, 3], N_max=100)
test_analyzer.time_series([1, 3], limits=True)
test_analyzer.phase_portrait(1, 3, analytical=True) 

plt.show()



# This part tests a grid of parameters
n=1
y00 = jnp.array([0])
y01 = jnp.array([1])
y02 = jnp.array([0])
y03 = jnp.array([1])
y0b = jnp.concatenate([y00, y01, y02, y03])
y0 = jnp.tile(y0b, n)

start_time = time.time()

omegas = jnp.geomspace(0.1, 10, 100)
r0s = jnp.linspace(0.1, 1, 10) 
r1s = jnp.linspace(0, 0, 1)
test_array = AdimBeamSystemArray(omegas, r0s, r1s)
test_array.solve(y0)
test_array.solve_ffts([1, 3], plot_bool=False, N_max=100)
test_array.vary_param(A_vs_Omega, [1, 3], "omega", "r0")

end_time = time.time()
print(end_time - start_time)
plt.show()
'''