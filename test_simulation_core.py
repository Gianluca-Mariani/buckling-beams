import jax.numpy as jnp
from adim_beams import phi
from adim_beams import lagrangian
from adim_beams import ODEs
from adim_beams import solve_system


def test_phi():
 # Test phi for various inputs
    assert jnp.isclose(phi(0), 0.0, atol=1e-2), "phi(0) should be 0"
    assert jnp.isclose(phi(1), jnp.pi, atol=1e-2), "phi(1) should be pi"
    assert jnp.isclose(phi(2), jnp.pi, atol=1e-2), "phi(2) should be pi"
    assert jnp.isclose(phi(3), 0.0, atol=1e-2), "phi(3) should be 0"
    assert jnp.isclose(phi(4), 0.0, atol=1e-2), "phi(4) should be 0"
    assert jnp.isclose(phi(5), jnp.pi, atol=1e-2), "phi(5) should be pi"
    assert jnp.isclose(phi(1000), 0.0, atol=1e-2), "phi(1000) should be 0"
    assert jnp.isclose(phi(2001), jnp.pi, atol=1e-2), "phi(2001) should be pi"

def test_lagrangian():
    q = jnp.array([0.1, 0.2, 0.3])  # Example displacement array
    t = 0.5  # Time
    result = lagrangian(q, t, omega=1.0, r0=0.5, r1=0.5)
    assert isinstance(result, jnp.ndarray), "Lagrangian should return a jnp array"
    assert result.shape == (), "Lagrangian should return a scalar value"

def test_ODEs():
    q = jnp.array([0.1, 0.2, 0.3])
    t = 0.5
    omega, r0, r1 = 1.0, 0.5, 0.5
    result = ODEs(t, q, omega, r0, r1)
    assert isinstance(result, jnp.ndarray), "ODEs should return a jnp array"
    assert result.shape == q.shape, "ODEs should return an array of the same shape as q"

def test_solve():
    omega, r0, r1 = 1.0, 0.5, 0.5
    y0 = jnp.array([0.1, 0.2, 0.3])  # Initial conditions
    sol = solve_system(y0, omega, r0, r1)
    # Test that the solution has a 'ts' (time) attribute
    assert hasattr(sol, 'ts'), "Solution should have 'ts' attribute"
    assert hasattr(sol, 'ys'), "Solution should have 'ys' attribute"
    # Check if the solution contains data points
    assert sol.ts.size > 0, "Solution time array should not be empty"
    assert sol.ys.shape[0] == sol.ts.size, "Solution ys should match the time array length"

def test_time_evolution():
    omega, r0, r1 = 1.0, 0.5, 0.5
    y0 = jnp.array([0.1, 0.2, 0.3])
    t_cycles = 5
    sol = solve_system(y0, omega, r0, r1, t_cycles=t_cycles)
    assert sol.ts[0] == 0.0, "Time should start at 0.0"
    assert sol.ts[-1] == t_cycles * 2 * jnp.pi / omega, f"Time should end at {t_cycles * jnp.pi / omega}"

def test_parameter_sensitivity():
    y0 = jnp.array([0.1, 0.2, 0.3])
    sol1 = solve_system(y0, omega=1.0, r0=0.5, r1=0.5)
    sol2 = solve_system(y0, omega=2.0, r0=0.5, r1=0.5)
    sol3 = solve_system(y0, omega=1.0, r0=0.8, r1=0.5)
    
    assert not jnp.allclose(sol1.ys, sol2.ys), "Solution should differ for different omega"
    assert not jnp.allclose(sol1.ys, sol3.ys), "Solution should differ for different r0"