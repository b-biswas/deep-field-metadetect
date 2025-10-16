from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp


@partial(jax.jit, static_argnames=["window_size"])
def local_maxima_filter(
    image: jnp.ndarray, window_size: int = 5, threshold: float = 0.0
) -> jnp.ndarray:
    """
    Find local maxima in an image using morphological operations.

    Parameters:
    -----------
    image : jnp.ndarray
        2D Input galaxy field
    window_size : int
        Size of the neighborhood for local maximum detection

    Returns:
    --------
    jnp.ndarray
        Binary mask indicating local maxima positions
    """
    pad_size = window_size // 2
    padded_image = jnp.pad(image, pad_size, mode="constant", constant_values=-jnp.inf)

    def is_local_max(i, j):
        center_val = padded_image[i + pad_size, j + pad_size]

        neighborhood = jax.lax.dynamic_slice(
            padded_image, (i, j), (window_size, window_size)
        )

        return (
            jnp.all(center_val >= neighborhood)
            & (jnp.sum(center_val == neighborhood) == 1)
        ) & (threshold <= center_val)

    height, width = image.shape
    i_indices, j_indices = jnp.meshgrid(
        jnp.arange(height), jnp.arange(width), indexing="ij"
    )

    local_max_mask = jax.vmap(
        jax.vmap(
            is_local_max,
            in_axes=(0, 0),
        ),
        in_axes=(0, 0),
    )(i_indices, j_indices)

    return local_max_mask


@partial(jax.jit, static_argnames=["window_size", "max_objects"])
def peak_finder(
    image: jnp.ndarray,
    threshold: float = 0.1,
    window_size: int = 5,
    max_objects: int = 100,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Find peaks in an image above a threshold with minimum separation.
    This function is not JITable because it uses jnp.argwhere

    Parameters:
    -----------
    image : jnp.ndarray
        2D Input galaxy field
    threshold : float
        Minimum pixel value all pixels in the window must satisfy
    window_size : int
        Size of the neighborhood for local maximum detection
    max_objects : int
        Maximum number of objects to detect (to make functions jitable)

    Returns:
    --------
    positions : jnp.ndarray
        Array of peak coordinates (y, x) of shape (max_objects, 2)
        Invalid entries filled with (-1, -1)
    """
    local_max_mask = local_maxima_filter(
        image, window_size=window_size, threshold=threshold
    )

    positions = jnp.argwhere(local_max_mask, size=max_objects, fill_value=(-1, -1))

    return positions


@partial(jax.jit, static_argnames=["window_size"])
def refine_centroid(
    image: jnp.ndarray, peak: Tuple[int, int], window_size: int = 5
) -> Tuple[float, float, bool]:
    """
    Refine peak position using intensity-weighted centroid.
    Skips refinement for objects too close to the border.
    Returns whether object was near border for warning purposes.

    Parameters:
    -----------
    image : jnp.ndarray
        2D Input galaxy field
    peak: jnp.ndarray
        Initial peak position
    window_size : int
        Size of window around peak for centroid calculation
        if window crosses image boudary, optimization is skipped.

    Returns:
    --------
    jnp.ndarray
        Refined peak coordinates (refined_y, refined_x) : float
        Note: original coordinatesare returned if near border
    near_border : bool
        True if object was near border and refinement was skipped
    """
    half_window = window_size // 2
    height, width = image.shape

    # If near border, return original coordinates
    near_border = (
        (peak[0] < half_window)
        | (peak[0] >= height - half_window)
        | (peak[1] < half_window)
        | (peak[1] >= width - half_window)
    )

    def border_case():
        return jnp.array([peak[0], peak[1]]).astype(float)

    def normal_case():
        window = jax.lax.dynamic_slice(
            image,
            (peak[0] - half_window, peak[1] - half_window),
            (window_size, window_size),
        )

        y_start = -half_window
        x_start = -half_window
        y_coords = jnp.arange(y_start, y_start + window_size)
        x_coords = jnp.arange(x_start, x_start + window_size)
        y_grid, x_grid = jnp.meshgrid(y_coords, x_coords, indexing="ij")

        total_intensity = jnp.sum(window)

        y_shift = jnp.sum((y_grid) * window) / total_intensity
        x_shift = jnp.sum((x_grid) * window) / total_intensity

        refined_y = y_shift + peak[0]
        refined_x = x_shift + peak[1]

        return jnp.array([refined_y, refined_x])

    result = jax.lax.cond(near_border, border_case, normal_case)

    return jnp.array([result[0], result[1]]), near_border


@partial(jax.jit, static_argnames=["window_size"])
def refine_centroid_in_cell(
    image: jnp.ndarray,
    peak_positions: jnp.ndarray,
    window_size: int = 5,
):
    """
    vmapped version of refine_centroid
    """
    return jax.vmap(refine_centroid, in_axes=(None, 0, None))(
        image, peak_positions, window_size
    )


@partial(jax.jit, static_argnames=["window_size", "refine_centroids", "max_objects"])
def detect_galaxies(
    image: jnp.ndarray,
    threshold: float = 0.0,
    window_size: int = 5,
    refine_centroids: bool = True,
    max_objects: int = 100,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Complete galaxy center detection pipeline with JIT compilation support.

    Parameters:
    -----------
    image : jnp.ndarray
        2D Input galaxy field
    threshold : float
        Minimum pixel value all pixels in the window must satisfy
    window_size : int
        Minimum distance between detected peaks
    refine_centroids : bool
        Whether to refine peak positions using centroid calculation
    max_objects : int
        Maximum number of objects to detect (for fixed array sizes)

    Returns:
    --------
    centers : jnp.ndarray
        Array of detected galaxy centers (y, x) of shape (max_objects, 2)
        Invalid entries filled with -1
    intensities : jnp.ndarray
        Array of peak intensities of shape (max_objects,)
        Invalid entries filled with 0
    border_flags : jnp.ndarray
        Array indicating which objects were near border (shape max_objects,)
    """
    peak_positions = peak_finder(image, threshold, window_size, max_objects)

    if not refine_centroids:
        border_flags = jnp.zeros(max_objects, dtype=bool)
        return peak_positions, peak_positions.astype(float), border_flags

    refined_positions, border_flags = refine_centroid_in_cell(
        image, peak_positions, window_size=5
    )

    return peak_positions, refined_positions, border_flags
