import jax.numpy as jnp
from adim_beams import BeamAnalyzer, AdimBeamSystemArray
from adim_beams import solve_system, A_vs_Omega
import matplotlib.pyplot as plt
import time

"""
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

"""
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