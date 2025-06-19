import jax.numpy as jnp
import numpy as np
from adim_beams import BeamAnalyzer, AdimBeamSystemArray, get_equilibria, rotate_solutions
from adim_beams import solve_system, A_vs_Omega
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import time
'''
from julia import Main
Main.include("find_equilibria.jl")
MyJulia = Main.FindEquilibria
n = 4
omega = 1.0
T = 2*np.pi / omega
times = np.linspace(0.0, T, 50)
r0 = 0.8
r1 = 0.1
aligned, real_result, stable_real_result = MyJulia.get_solutions_flags(n, times, omega, r0, r1)
aligned_np = np.swapaxes(np.array(aligned), 0, 1)
real_result_np = np.swapaxes(np.array(real_result), 0, 1)
stable_real_result_np = np.swapaxes(np.array(stable_real_result), 0, 1)
unstable_real_result_np = real_result_np & ~stable_real_result_np

fig0, ax0 = plt.subplots()
fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
fig3, ax3 = plt.subplots()
fig8, ax8 = plt.subplots()

l1 = 0
l2 = 0

for (i, sol) in enumerate(aligned_np):
    masked_sol = np.real(sol[stable_real_result_np[i]])
    masked_times = times[stable_real_result_np[i]]

    if len(masked_sol) > 0:
        ax0.plot(masked_times, masked_sol[:,0], marker='o', linestyle='None', markersize=5)
        ax1.plot(masked_times, masked_sol[:,1], marker='o', linestyle='None', markersize=5)
        ax2.plot(masked_times, masked_sol[:,2], marker='o', linestyle='None', markersize=5)
        ax3.plot(masked_times, masked_sol[:,3], marker='o', linestyle='None', markersize=5)
        ax8.plot(masked_sol[:,0], masked_sol[:,1], linestyle='-', linewidth=3)
        l1 += 1

    unstable_sol = np.real(sol[unstable_real_result_np[i]])
    unstable_times = times[unstable_real_result_np[i]]

    if len(unstable_sol) > 0 and False:
        ax0.plot(unstable_times, unstable_sol[:,0], marker='x', linestyle='None', markersize=1)
        ax1.plot(unstable_times, unstable_sol[:,1], marker='x', linestyle='None', markersize=1)
        ax2.plot(unstable_times, unstable_sol[:,2], marker='x', linestyle='None', markersize=1)
        ax3.plot(unstable_times, unstable_sol[:,3], marker='x', linestyle='None', markersize=1)
        ax8.plot(unstable_sol[:,0], unstable_sol[:,1], linestyle=':', linewidth=1)
        l2 += 1

print(l1)

plt.show()
'''

# This part tests a single combination of parameters
n=1
y00 = jnp.array([0])
y01 = jnp.array([0.7])
y02 = jnp.array([0])
y03 = jnp.array([0.7])
y04 = jnp.array([0])
y05 = jnp.array([-0.7])
y06 = jnp.array([0])
y07 = jnp.array([-0.7])
y0b = jnp.concatenate([y00, y01, y02, y03, y04, y05, y06, y07])
#y0b = jnp.concatenate([y00, y01, y02, y03])
y0 = jnp.tile(y0b, n)

omega = 0.1
r0=0.4
r1=0.5 

test_sol = solve_system(y0, omega=omega, r0=r0, r1=r1, t_cycles=3, N_fact=2000)
stable_part_rotated, unstable_part_rotated, stable_times, unstable_times, numerical_rotated = get_equilibria(len(y0), np.array(test_sol.ts, dtype=np.float64), omega, r0, r1, test_sol.ys, rotate=False)
test_analyzer = BeamAnalyzer(test_sol.ts, numerical_rotated, omega=omega, r0=r0, r1=r1)
test_analyzer.time_series([0, 1, 2, 3, 4, 5, 6, 7], equilibria=True, unstable=False, stable_sol = stable_part_rotated, unstable_sol = unstable_part_rotated, stable_time = stable_times, unstable_time = unstable_times)
test_analyzer.phase_portrait(0, 1, equilibria=True, unstable=False, stable_sol = stable_part_rotated, unstable_sol = unstable_part_rotated)
plt.show()

'''
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

