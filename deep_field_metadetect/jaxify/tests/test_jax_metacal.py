import multiprocessing

import jax.numpy as jnp
import numpy as np
import pytest

from deep_field_metadetect.jaxify.jax_metacal import (
    jax_metacal_op_shears,
)
from deep_field_metadetect.jaxify.observation import (
    ngmix_obs_to_dfmd_obs,
)
from deep_field_metadetect.metacal import metacal_op_shears
from deep_field_metadetect.utils import (
    assert_m_c_ok,
    estimate_m_and_c,
    fit_gauss_mom_mcal_res,
    make_simple_sim,
    measure_mcal_shear_quants,
    print_m_c,
)


def _run_single_sim_pair(seed, s2n):
    nxy = 53
    nxy_psf = 53
    scale = 0.2
    obs_plus, *_ = make_simple_sim(
        seed=seed,
        g1=0.02,
        g2=0.0,
        s2n=s2n,
        dim=nxy,
        dim_psf=nxy_psf,
        scale=scale,
        deep_noise_fac=1.0 / np.sqrt(10),
        deep_psf_fac=1.0,
        return_dfmd_obs=True,
    )
    mcal_res = jax_metacal_op_shears(
        obs_plus,
        nxy_psf=nxy_psf,
        scale=scale,
    )
    res_p = fit_gauss_mom_mcal_res(mcal_res)
    res_p = measure_mcal_shear_quants(res_p)

    obs_minus, *_ = make_simple_sim(
        seed=seed,
        g1=-0.02,
        g2=0.0,
        s2n=s2n,
        dim=nxy,
        dim_psf=nxy_psf,
        scale=scale,
        deep_noise_fac=1.0 / np.sqrt(10),
        deep_psf_fac=1.0,
        return_dfmd_obs=True,
    )
    mcal_res = jax_metacal_op_shears(
        obs_minus,
        nxy_psf=nxy_psf,
        scale=scale,
    )
    res_m = fit_gauss_mom_mcal_res(mcal_res)
    res_m = measure_mcal_shear_quants(res_m)

    return res_p, res_m


def _run_single_sim_pair_jax_and_ngmix(seed, s2n):
    nxy = 53
    nxy_psf = 53
    scale = 0.2
    obs_plus, *_ = make_simple_sim(
        seed=seed,
        g1=0.02,
        g2=0.0,
        s2n=s2n,
        dim=nxy,
        dim_psf=nxy_psf,
        scale=scale,
        deep_noise_fac=1.0 / np.sqrt(10),
        deep_psf_fac=1.0,
        return_dfmd_obs=False,
    )

    mcal_res_ngmix = metacal_op_shears(obs_plus)

    res_p_ngmix = fit_gauss_mom_mcal_res(mcal_res_ngmix)
    res_p_ngmix = measure_mcal_shear_quants(res_p_ngmix)

    obs_plus = ngmix_obs_to_dfmd_obs(obs_plus)

    mcal_res = jax_metacal_op_shears(
        obs_plus,
        nxy_psf=nxy_psf,
        scale=scale,
    )
    res_p = fit_gauss_mom_mcal_res(mcal_res)
    res_p = measure_mcal_shear_quants(res_p)

    obs_minus, *_ = make_simple_sim(
        seed=seed,
        g1=-0.02,
        g2=0.0,
        s2n=s2n,
        dim=nxy,
        dim_psf=nxy_psf,
        scale=scale,
        deep_noise_fac=1.0 / np.sqrt(10),
        deep_psf_fac=1.0,
        return_dfmd_obs=False,
    )

    mcal_res_ngmix = metacal_op_shears(obs_minus)
    res_m_ngmix = fit_gauss_mom_mcal_res(mcal_res_ngmix)
    res_m_ngmix = measure_mcal_shear_quants(res_m_ngmix)

    obs_minus = ngmix_obs_to_dfmd_obs(obs_minus)
    mcal_res = jax_metacal_op_shears(
        obs_minus,
        nxy_psf=nxy_psf,
        scale=scale,
    )
    res_m = fit_gauss_mom_mcal_res(mcal_res)
    res_m = measure_mcal_shear_quants(res_m)

    return (res_p, res_m), (res_p_ngmix, res_m_ngmix)


def test_metacal_smoke():
    res_p, res_m = _run_single_sim_pair(1234, 1e8)
    for col in res_p.dtype.names:
        assert np.isfinite(res_p[col]).all()
        assert np.isfinite(res_m[col]).all()


def test_metacal_jax_vs_ngmix():
    nsims = 5

    rng = np.random.RandomState(seed=34132)
    seeds = rng.randint(size=nsims, low=1, high=2**29)
    res_p = []
    res_m = []
    res_p_ngmix = []
    res_m_ngmix = []
    for seed in seeds:
        res, res_ngmix = _run_single_sim_pair_jax_and_ngmix(seed, 1e8)
        if res is not None:
            res_p.append(res[0])
            res_m.append(res[1])

            res_p_ngmix.append(res_ngmix[0])
            res_m_ngmix.append(res_ngmix[1])

            assert np.allclose(
                res[0].tolist(),
                res_ngmix[0].tolist(),
                atol=1e-3,
                rtol=0.01,
                equal_nan=True,
            )
            assert np.allclose(
                res[1].tolist(),
                res_ngmix[1].tolist(),
                atol=1e-3,
                rtol=0.01,
                equal_nan=True,
            )

    m, merr, c1, c1err, c2, c2err = estimate_m_and_c(
        np.concatenate(res_p),
        np.concatenate(res_m),
        0.02,
        jackknife=len(res_p),
    )

    m_ng, merr_ng, c1_ng, c1err_ng, c2_ng, c2err_ng = estimate_m_and_c(
        np.concatenate(res_p_ngmix),
        np.concatenate(res_m_ngmix),
        0.02,
        jackknife=len(res_p_ngmix),
    )

    print("JAX results:")
    print_m_c(m, merr, c1, c1err, c2, c2err)
    print("ngmix results:")
    print_m_c(m_ng, merr_ng, c1_ng, c1err_ng, c2_ng, c2err_ng)
    assert_m_c_ok(m, merr, c1, c1err, c2, c2err)

    assert np.allclose(m, m_ng, atol=1e-4)
    assert np.allclose(merr, merr_ng, atol=1e-4)
    assert np.allclose(c1err, c1err_ng, atol=1e-6)
    assert np.allclose(c1, c1_ng, atol=1e-6)
    assert np.allclose(c2err, c2err_ng, atol=1e-6)
    assert np.allclose(c2, c2_ng, atol=1e-6)


def test_metacal():
    nsims = 50

    rng = np.random.RandomState(seed=34132)
    seeds = rng.randint(size=nsims, low=1, high=2**29)
    res_p = []
    res_m = []
    for seed in seeds:
        res = _run_single_sim_pair(seed, 1e8)
        if res is not None:
            res_p.append(res[0])
            res_m.append(res[1])

    m, merr, c1, c1err, c2, c2err = estimate_m_and_c(
        np.concatenate(res_p),
        np.concatenate(res_m),
        0.02,
        jackknife=len(res_p),
    )

    print_m_c(m, merr, c1, c1err, c2, c2err)
    assert_m_c_ok(m, merr, c1, c1err, c2, c2err)


@pytest.mark.slow
def test_metacal_slow():  # pragma: no cover
    nsims = 100_000
    chunk_size = multiprocessing.cpu_count() * 100
    nchunks = nsims // chunk_size + 1
    nsims = nchunks * chunk_size

    rng = np.random.RandomState(seed=34132)
    seeds = rng.randint(size=nsims, low=1, high=2**29)
    res_p = []
    res_m = []
    loc = 0
    for chunk in range(nchunks):
        _seeds = seeds[loc : loc + chunk_size]
        for seed in _seeds:
            res = _run_single_sim_pair(seed, 20)
            if res is not None:
                res_p.append(res[0])
                res_m.append(res[1])

        if len(res_p) < 500:
            njack = len(res_p)
        else:
            njack = 100

        m, merr, c1, c1err, c2, c2err = estimate_m_and_c(
            np.concatenate(res_p),
            np.concatenate(res_m),
            0.02,
            jackknife=njack,
        )

        print("# of sims:", len(res_p), flush=True)
        print_m_c(m, merr, c1, c1err, c2, c2err)

        loc += chunk_size

    print_m_c(m, merr, c1, c1err, c2, c2err)
    assert_m_c_ok(m, merr, c1, c1err, c2, c2err)


def test_jax_vs_ngmix_render_psf_and_build_obs():
    """Test _jax_render_psf_and_build_obs vs render_psf_and_build_obs"""
    import galsim
    import jax_galsim

    from deep_field_metadetect.jaxify.jax_metacal import (
        _jax_render_psf_and_build_obs,
        jax_get_gauss_reconv_psf_galsim,
    )
    from deep_field_metadetect.jaxify.jax_utils import compute_stepk
    from deep_field_metadetect.jaxify.observation import ngmix_obs_to_dfmd_obs
    from deep_field_metadetect.metacal import (
        _render_psf_and_build_obs,
        get_gauss_reconv_psf_galsim,
    )
    from deep_field_metadetect.utils import make_simple_sim

    # Create test observations
    nxy = 53
    nxy_psf = 21
    scale = 0.2

    ngmix_obs, _, _ = make_simple_sim(
        seed=12345,
        g1=0.0,
        g2=0.0,
        s2n=1e8,
        dim=nxy,
        dim_psf=nxy_psf,
        scale=scale,
        return_dfmd_obs=False,
    )

    # Convert to dfmd observation
    dfmd_obs = ngmix_obs_to_dfmd_obs(ngmix_obs)

    # Create test reconv PSFs
    test_image = jnp.ones((nxy, nxy))

    # JAX version
    jax_psf = jax_galsim.Gaussian(sigma=1.0).withFlux(1.0)
    dk = compute_stepk(pixel_scale=scale, image_size=nxy_psf)
    jax_reconv_psf = jax_get_gauss_reconv_psf_galsim(
        jax_psf, dk=dk, nxy_psf=nxy_psf, kim_size=256
    )
    jax_result = _jax_render_psf_and_build_obs(
        test_image, dfmd_obs, jax_reconv_psf, nxy_psf=nxy_psf, weight_fac=1
    )

    # ngmix version
    ngmix_psf = galsim.Gaussian(sigma=1.0).withFlux(1.0)
    ngmix_reconv_psf = get_gauss_reconv_psf_galsim(ngmix_psf, dk=dk, kim_size=256)

    ngmix_result = _render_psf_and_build_obs(
        test_image, ngmix_obs, ngmix_reconv_psf, weight_fac=1
    )

    assert jnp.isclose(ngmix_reconv_psf.sigma, jax_reconv_psf.sigma), (
        "reconv psf sigmas are different"
    )
    # Check if shapes match
    assert jax_result.psf.image.shape == ngmix_result.psf.image.shape, (
        f"PSF shapes don't match: JAX {jax_result.psf.image.shape} "
        f"vs ngmix {ngmix_result.psf.image.shape}"
    )

    # Compare PSF images with some tolerance
    diff = jnp.abs(jax_result.psf.image - ngmix_result.psf.image)
    max_diff = jnp.max(diff)
    rel_diff = max_diff / jnp.max(jax_result.psf.image)

    print(f"Max absolute difference: {max_diff}")
    print(f"Max relative difference: {rel_diff}")

    # Test that PSF images are reasonably close
    assert jnp.allclose(
        jax_result.psf.image, ngmix_result.psf.image, atol=1e-10, rtol=1e-6
    ), f"PSF images differ significantly. Max diff: {max_diff}, Rel diff: {rel_diff}"
