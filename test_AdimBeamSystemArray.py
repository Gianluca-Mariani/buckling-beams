import jax.numpy as jnp
from adim_beams import AdimBeamSystemArray

def test_initialization():
    omegas = jnp.array([1.0, 2.0, 3.0])
    r0s = jnp.array([0.5, 0.6])
    r1s = jnp.array([0.3, 0.4])
    system = AdimBeamSystemArray(omegas=omegas, r0s=r0s, r1s=r1s)
    assert system.lengths.shape == (3,), "lengths should be a 1D array"
    assert system.omegas.shape == (len(omegas), len(r0s), len(r1s)), "omega should match input shape"
    assert system.r0s.shape == (len(omegas), len(r0s), len(r1s)), "r0 should match input shape"
    assert system.r1s.shape == (len(omegas), len(r0s), len(r1s)), "r1 should match input shape"
    actual_shape = (
    len(system.beams_param),
    len(system.beams_param[0]),
    len(system.beams_param[0][0]),
    )
    expected_shape = (len(omegas), len(r0s), len(r1s))
    assert actual_shape == expected_shape, f"beams_param shape mismatch: got {actual_shape}, expected {expected_shape}"
    assert (system.omegas[0, :, :] == 1.0).all(), "omega[0] should be 1.0"
    assert (system.r0s[:, 0, :] == 0.5).all(), "r0[0] should be 0.5"
    assert (system.r1s[:, :, 0] == 0.3).all(), "r1[0] should be 0.3"
    assert (system.omegas[1, :, :] == 2.0).all(), "omega[1] should be 2.0"
    assert (system.r0s[:, 1, :] == 0.6).all(), "r0[1] should be 0.6"
    assert (system.r1s[:, :, 1] == 0.4).all(), "r1[1] should be 0.4"
    assert (system.omegas[2, :, :] == 3.0).all(), "omega[1] should be 3.0"
    assert system.beams_param[0][1][0].omega == 1.0, "omega[0] should be 1.0"
