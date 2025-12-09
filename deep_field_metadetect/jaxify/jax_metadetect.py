import ngmix
import numpy as np

from deep_field_metadetect.detect import (
    generate_mbobs_for_detections,
    run_detection_sep,
)
from deep_field_metadetect.jaxify.jax_metacal import (
    DEFAULT_FFT_SIZE,
    DEFAULT_SHEARS,
    DEFAULT_STEP,
    jax_metacal_wide_and_deep_psf_matched,
)
from deep_field_metadetect.mfrac import compute_mfrac_interp_image
from deep_field_metadetect.utils import fit_gauss_mom_obs, fit_gauss_mom_obs_and_psf


def jax_single_band_deep_field_metadetect(
    obs_wide,
    obs_deep,
    obs_deep_noise,
    nxy,
    nxy_psf,
    step=DEFAULT_STEP,
    shears=None,
    skip_obs_wide_corrections=False,
    skip_obs_deep_corrections=False,
    nodet_flags=0,
    scale=0.2,
    return_k_info=False,
    force_stepk_field=0.0,
    force_maxk_field=0.0,
    force_stepk_psf=0.0,
    force_maxk_psf=0.0,
    fft_size=DEFAULT_FFT_SIZE,
) -> dict:
    """Run deep-field metadetection for a simple scenario of a single band
    with a single image per band using only post-PSF Gaussian weighted moments.

    Parameters
    ----------
    obs_wide : DFMdetObservation
        The wide-field observation.
    obs_deep : DFMdetObservation
        The deep-field observation.
    obs_deep_noise : DFMdetObservation
        The deep-field noise observation.
    nxy: int
        Image size
    nxy_psf: int
        PSF size
    step : float, optional
        The step size for the metacalibration, by default DEFAULT_STEP.
    shears : list, optional
        The shears to use for the metacalibration, by default DEFAULT_SHEARS
        if set to None.
    skip_obs_wide_corrections : bool, optional
        Skip the observation corrections for the wide-field observations,
        by default False.
    skip_obs_deep_corrections : bool, optional
        Skip the observation corrections for the deep-field observations,
        by default False.
    nodet_flags : int, optional
        The bmask flags marking area in the image to skip, by default 0.
    scale: float
        pixel scale

    Returns
    -------
    dfmdet_res : numpy.ndarray
        The deep-field metadetection results as a structured array containing
        detection and measurement results for all shears.

    Note: If return_k_info is set to True for debugging,
    this function returns a dict containing dfmdet_res and kinfo. kinfo being:
    (_force_stepk_field, _force_maxk_field, _force_stepk_psf, _force_maxk_psf)
    """
    if shears is None:
        shears = DEFAULT_SHEARS

    mcal_res = jax_metacal_wide_and_deep_psf_matched(
        obs_wide=obs_wide,
        obs_deep=obs_deep,
        obs_deep_noise=obs_deep_noise,
        nxy=nxy,
        nxy_psf=nxy_psf,
        step=step,
        shears=shears,
        skip_obs_wide_corrections=skip_obs_wide_corrections,
        skip_obs_deep_corrections=skip_obs_deep_corrections,
        scale=scale,
        return_k_info=return_k_info,
        force_stepk_field=force_stepk_field,
        force_maxk_field=force_maxk_field,
        force_stepk_psf=force_stepk_psf,
        force_maxk_psf=force_maxk_psf,
        fft_size=fft_size,
    )  # This returns ngmix Obs for now

    psf_res = fit_gauss_mom_obs(mcal_res["noshear"].psf)
    dfmdet_res = []
    for shear in shears:
        obs = mcal_res[shear]
        detres = run_detection_sep(obs, nodet_flags=nodet_flags)

        ixc = (detres["catalog"]["x"] + 0.5).astype(int)
        iyc = (detres["catalog"]["y"] + 0.5).astype(int)
        bmask_flags = obs.bmask[iyc, ixc]

        mfrac_vals = np.zeros_like(bmask_flags, dtype="f4")
        if np.any(obs.mfrac > 0):
            _interp_mfrac = compute_mfrac_interp_image(
                obs.mfrac,
                obs.jacobian.get_galsim_wcs(),
            )
            for i, (x, y) in enumerate(
                zip(detres["catalog"]["x"], detres["catalog"]["y"])
            ):
                mfrac_vals[i] = _interp_mfrac.xValue(x, y)

        for ind, (obj, mbobs) in enumerate(
            generate_mbobs_for_detections(
                ngmix.observation.get_mb_obs(obs),
                xs=detres["catalog"]["x"],
                ys=detres["catalog"]["y"],
            )
        ):
            fres = fit_gauss_mom_obs_and_psf(mbobs[0][0], psf_res=psf_res)
            dfmdet_res.append(
                (ind + 1, obj["x"], obj["y"], shear, bmask_flags[ind], mfrac_vals[ind])
                + tuple(fres[0])
            )

    total_dtype = [
        ("id", "i8"),
        ("x", "f8"),
        ("y", "f8"),
        ("mdet_step", "U7"),
        ("bmask_flags", "i4"),
        ("mfrac", "f4"),
    ] + fres.dtype.descr

    if return_k_info:
        result = {
            "mdetect_res": np.array(dfmdet_res, dtype=total_dtype),
            "kinfo": mcal_res.get("kinfo") if return_k_info else None,
        }
        return result

    return np.array(dfmdet_res, dtype=total_dtype)
