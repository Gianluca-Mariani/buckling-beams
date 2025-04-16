import jax.numpy as jnp
from adim_beams import AdimBeamSystemArray


def test_initialization():
    omegas = jnp.array([1.0, 2.0, 3.0])
    r0s = jnp.array([0.5, 0.6])
    r1s = jnp.array([0.3, 0.4])
    system = AdimBeamSystemArray(omegas=omegas, r0s=r0s, r1s=r1s)
    assert system.lengths.shape == (3,), "lengths should be a 1D array"
    assert system.omegas.shape == (
        len(omegas),
        len(r0s),
        len(r1s),
    ), "omega should match input shape"
    assert system.r0s.shape == (
        len(omegas),
        len(r0s),
        len(r1s),
    ), "r0 should match input shape"
    assert system.r1s.shape == (
        len(omegas),
        len(r0s),
        len(r1s),
    ), "r1 should match input shape"
    assert (system.omegas[0, :, :] == 1.0).all(), "omega[0] should be 1.0"
    assert (system.r0s[:, 0, :] == 0.5).all(), "r0[0] should be 0.5"
    assert (system.r1s[:, :, 0] == 0.3).all(), "r1[0] should be 0.3"
    assert (system.omegas[1, :, :] == 2.0).all(), "omega[1] should be 2.0"
    assert (system.r0s[:, 1, :] == 0.6).all(), "r0[1] should be 0.6"
    assert (system.r1s[:, :, 1] == 0.4).all(), "r1[1] should be 0.4"
    assert (system.omegas[2, :, :] == 3.0).all(), "omega[1] should be 3.0"


def test_solve():
    omegas = jnp.array([1.0, 2.0])
    r0s = jnp.array([0.5, 0.6, 0.7])
    r1s = jnp.array([0.3, 0.4, 0.5, 0.6])
    system = AdimBeamSystemArray(omegas=omegas, r0s=r0s, r1s=r1s)

    y0 = jnp.array([0.1, 0.2, 0.3, 0.4])
    t_cycles = 6
    N_fact = 1000
    system.solve(y0=y0, t_cycles=t_cycles, N_fact=N_fact)

    assert hasattr(system, "ts"), "System should have 'ts' attribute after solving"
    assert hasattr(system, "ys"), "System should have 'ys' attribute after solving"
    assert system.ts.shape == tuple(system.lengths) + (
        N_fact,
    ), "Time array should match expected length"
    assert system.ys.shape == tuple(system.lengths) + (
        N_fact,
        len(y0),
    ), "Solution array should match expected shape"


def test_solve_ffts():
    omegas = jnp.array([1.0, 2.0])
    r0s = jnp.array([0.5, 0.6, 0.7])
    r1s = jnp.array([0.3, 0.4, 0.5, 0.6])
    system = AdimBeamSystemArray(omegas=omegas, r0s=r0s, r1s=r1s)

    y0 = jnp.array([0.1, 0.2, 0.3, 0.4])
    t_cycles = 8
    N_fact = 3000
    system.solve(y0=y0, t_cycles=t_cycles, N_fact=N_fact)

    system.solve_ffts([0, 1, 2, 3], plot_bool=False, N_max=100)

    assert hasattr(system, "xf"), "System should have 'xf' attribute after solving FFT"
    assert hasattr(
        system, "fft_results"
    ), "System should have 'yf' attribute after solving FFT"
    assert hasattr(
        system, "dominant_frequencies"
    ), "System should have 'freqs' attribute after solving FFT"
    assert hasattr(
        system, "dominant_amplitudes"
    ), "System should have 'amps' attribute after solving FFT"
    assert system.xf.shape == tuple(system.lengths) + (
        N_fact // 2,
    ), "Frequency array should match expected length"
    assert system.fft_results.shape == tuple(system.lengths) + (
        len(y0),
        N_fact,
    ), "FFT results should match expected shape"
    assert system.dominant_frequencies.shape == tuple(system.lengths) + (
        len(y0),
    ), "Dominant frequencies should match expected shape"
    assert system.dominant_amplitudes.shape == tuple(system.lengths) + (
        len(y0),
    ), "Dominant frequencies should match expected shape"
