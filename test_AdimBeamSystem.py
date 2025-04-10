import jax.numpy as jnp
from adim_beams import AdimBeamSystem

def test_initialization():
    system = AdimBeamSystem(omega=1.0, r0=0.5, r1=0.3)
    assert system.omega == 1.0, "omega should be 1.0"
    assert system.r0 == 0.5, "r0 should be 0.5"
    assert system.r1 == 0.3, "r1 should be 0.3"
    assert system.rounding == 2, "rounding should be 2"

def test_phi():
    # Create an instance of AdimBeamSystem
    system = AdimBeamSystem(omega=1.0, r0=0.5, r1=0.5)

    # Test phi for various inputs
    assert jnp.isclose(system.phi(0), 0.0, atol=1e-2), "phi(0) should be 0"
    assert jnp.isclose(system.phi(1), jnp.pi, atol=1e-2), "phi(1) should be pi"
    assert jnp.isclose(system.phi(2), jnp.pi, atol=1e-2), "phi(2) should be pi"
    assert jnp.isclose(system.phi(3), 0.0, atol=1e-2), "phi(3) should be 0"
    assert jnp.isclose(system.phi(4), 0.0, atol=1e-2), "phi(4) should be 0"
    assert jnp.isclose(system.phi(5), jnp.pi, atol=1e-2), "phi(5) should be pi"
    assert jnp.isclose(system.phi(1000), 0.0, atol=1e-2), "phi(1000) should be 0"
    assert jnp.isclose(system.phi(2001), jnp.pi, atol=1e-2), "phi(2001) should be pi"

def test_lagrangian():
    system = AdimBeamSystem(omega=1.0, r0=0.5, r1=0.5)
    q = jnp.array([0.1, 0.2, 0.3])  # Example displacement array
    t = 0.5  # Time
    result = system.lagrangian(q, t, system.omega, system.r0, system.r1)
    assert isinstance(result, jnp.ndarray), "Lagrangian should return a jnp array"
    assert result.shape == (), "Lagrangian should return a scalar value"

def test_ODEs():
    system = AdimBeamSystem(omega=1.0, r0=0.5, r1=0.5)
    q = jnp.array([0.1, 0.2, 0.3])
    t = 0.5
    args = (system.omega, system.r0, system.r1)
    result = system.ODEs(t, q, args)
    assert isinstance(result, jnp.ndarray), "ODEs should return a jnp array"
    assert result.shape == q.shape, "ODEs should return an array of the same shape as q"

def test_solve():
    system = AdimBeamSystem(omega=1.0, r0=0.5, r1=0.5)
    y0 = jnp.array([0.1, 0.2, 0.3])  # Initial conditions
    sol = system.solve(y0, system.omega, system.r0, system.r1)
    # Test that the solution has a 'ts' (time) attribute
    assert hasattr(sol, 'ts'), "Solution should have 'ts' attribute"
    assert hasattr(sol, 'ys'), "Solution should have 'ys' attribute"
    # Check if the solution contains data points
    assert sol.ts.size > 0, "Solution time array should not be empty"
    assert sol.ys.shape[0] == sol.ts.size, "Solution ys should match the time array length"

def test_time_evolution():
    system = AdimBeamSystem(omega=1.0, r0=0.5, r1=0.5)
    y0 = jnp.array([0.1, 0.2, 0.3])
    t_cycles = 5
    sol = system.solve(y0, system.omega, system.r0, system.r1, t_cycles=t_cycles)
    assert sol.ts[0] == 0.0, "Time should start at 0.0"
    assert sol.ts[-1] == t_cycles * 2 * jnp.pi / system.omega, f"Time should end at {system.t_cycles * jnp.pi / system.omega}"

def test_parameter_sensitivity():
    system1 = AdimBeamSystem(omega=1.0, r0=0.5, r1=0.5)
    system2 = AdimBeamSystem(omega=2.0, r0=0.5, r1=0.5)
    system3 = AdimBeamSystem(omega=1.0, r0=0.8, r1=0.5)
    
    y0 = jnp.array([0.1, 0.2, 0.3])
    sol1 = system1.solve(y0, system1.omega, system1.r0, system1.r1)
    sol2 = system2.solve(y0, system2.omega, system2.r0, system2.r1)
    sol3 = system3.solve(y0, system3.omega, system3.r0, system3.r1)
    
    assert not jnp.allclose(sol1.ys, sol2.ys), "Solution should differ for different omega"
    assert not jnp.allclose(sol1.ys, sol3.ys), "Solution should differ for different r0"