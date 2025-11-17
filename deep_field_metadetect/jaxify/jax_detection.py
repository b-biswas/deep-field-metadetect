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
    threshold : float
        Minimum pixel value (absolute) a central pixel must satisfy

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

        return (jnp.all(center_val >= neighborhood)) & (threshold <= center_val)

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
        Minimum pixel value (absolute) a central pixel must satisfy
    window_size : int
        Size of the neighborhood for local maximum detection
    max_objects : int
        Maximum number of objects to detect (to make functions jitable)

    Returns:
    --------
    positions : jnp.ndarray
        Array of peak coordinates (y, x) of shape (max_objects, 2)
        Invalid entries filled with (-999, -999)
    """
    local_max_mask = local_maxima_filter(
        image, window_size=window_size, threshold=threshold
    )

    positions = jnp.argwhere(local_max_mask, size=max_objects, fill_value=(-999, -999))

    return positions


@partial(jax.jit, static_argnames=["window_size"])
def refine_centroid(
    image: jnp.ndarray, peak: Tuple[int, int], window_size: int = 5
) -> Tuple[float, float, bool]:
    """
    Refine peak position of single object using intensity-weighted centroid.
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
        Minimum pixel value (absolute) a central pixel must satisfy
    window_size : int
        Minimum distance between detected peaks
    refine_centroids : bool
        Whether to refine peak positions using centroid calculation
    max_objects : int
        Maximum number of objects to detect (for fixed array sizes)

    Returns:
    --------
    peak_positions : jnp.ndarray
        Array of detected galaxy centers (y, x) of shape (max_objects, 2).
        Returns only the integral pixel location.
        Invalid entries filled with -999
    refined_positions : jnp.ndarray
        Array of detected galaxy centers (y, x) after centroid refinement.
        Returns the refined floating point values of the center.
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


@partial(jax.jit, static_argnames=["max_iterations"])
def watershed_segmentation(
    image: jnp.ndarray,
    markers: jnp.ndarray,
    mask: jnp.ndarray = None,
    max_iterations: int = 30,
) -> jnp.ndarray:
    """
    JAX implementation of watershed segmentation algorithm.

    Parameters:
    -----------
    image : jnp.ndarray
        2D input image (typically distance transform or inverted intensity)
    markers : jnp.ndarray
        2D array of initial markers (labeled regions) where positive values
        indicate different watershed basins and 0 indicates unmarked pixels
    mask : jnp.ndarray, optional
        Binary mask indicating valid pixels for segmentation.
        Pixels with non-zero masked values are masked.
    max_iterations : int
        Maximum number of iterations for the flooding process

    Returns:
    --------
    labels : jnp.ndarray
        2D segmentation map with same shape as input image
    """
    if mask is None:
        mask = jnp.zeros_like(image, dtype=bool)

    labels = markers.copy()
    height, width = image.shape

    def watershed_step(labels_prev):
        """Single iteration of watershed flooding"""
        labels_new = labels_prev.copy()

        def update_pixel(i, j):
            # Skip if masked out
            # Note: another option here would be skip if already labeled
            current_label = labels_prev[i, j]
            is_valid = ~mask[i, j]

            def check_neighbors():
                # Check 4-connected neighbors
                neighbor_coords = jnp.array(
                    [[i - 1, j], [i + 1, j], [i, j - 1], [i, j + 1]]
                )

                in_bounds = (
                    (neighbor_coords[:, 0] >= 0)
                    & (neighbor_coords[:, 0] < height)
                    & (neighbor_coords[:, 1] >= 0)
                    & (neighbor_coords[:, 1] < width)
                )

                neighbor_labels = labels_prev[
                    neighbor_coords[:, 0], neighbor_coords[:, 1]
                ]
                neighbor_values = image[neighbor_coords[:, 0], neighbor_coords[:, 1]]

                # Mask for valid (labeled and in-bounds) neighbors
                valid_mask = in_bounds & (neighbor_labels > 0)

                has_valid = jnp.any(valid_mask)

                def process_valid_neighbors():
                    # Use large value for invalid neighbors in argmin
                    masked_values = jnp.where(valid_mask, neighbor_values, jnp.inf)
                    min_idx = jnp.argmin(masked_values)

                    # Check if current pixel should be flooded
                    current_value = image[i, j]
                    min_neighbor_value = neighbor_values[min_idx]

                    # Flood if current value is >= minimum neighbor value
                    def should_flood():
                        """Decides when to flood a pixel based on if it is marked"""

                        def unmarked_pixel():
                            return (current_value + 0.05) >= min_neighbor_value

                        def marked_pixel():
                            return (current_value) >= min_neighbor_value

                        is_marked = current_label != 0

                        return jax.lax.cond(
                            is_marked,
                            marked_pixel,
                            unmarked_pixel,
                        )

                    to_flood = should_flood()

                    # leave current value if update is not required
                    return jax.lax.cond(
                        to_flood,
                        lambda: neighbor_labels[min_idx],
                        lambda: current_label,
                    )

                # If no valid neighbors, leave current value else process
                return jax.lax.cond(
                    has_valid, process_valid_neighbors, lambda: current_label
                )

            # If pixel is not maked, check for neightbors
            new_label = jax.lax.cond(is_valid, check_neighbors, lambda: current_label)

            return new_label

        # Vectorized update over all pixels
        i_coords, j_coords = jnp.meshgrid(
            jnp.arange(height), jnp.arange(width), indexing="ij"
        )

        labels_new = jax.vmap(jax.vmap(update_pixel, in_axes=(0, 0)), in_axes=(0, 0))(
            i_coords, j_coords
        )

        return labels_new

    # Iterative flooding using scan
    def scan_fn(labels_current, _):
        labels_next = watershed_step(labels_current)
        return labels_next, None

    final_labels, _ = jax.lax.scan(scan_fn, labels, jnp.arange(max_iterations))

    return final_labels


@partial(jax.jit, static_argnames=["max_iterations"])
def watershed_from_peaks(
    image: jnp.ndarray,
    peaks: jnp.ndarray,
    mask: jnp.ndarray = None,
    max_iterations: int = 30,
) -> jnp.ndarray:
    """
    Perform watershed segmentation using detected peaks as markers.

    Parameters:
    -----------
    image : jnp.ndarray
        2D input image
    peaks : jnp.ndarray
        Array of peak positions (y, x) of shape (n_peaks, 2)
    mask : jnp.ndarray
        Array of masked pixels.
        Pixels with non-zero masked values are masked.
    max_iterations : int
        Maximum iterations for watershed algorithm

    Returns:
    --------
    watershed_labels : jnp.ndarray
        2D segmentation map from watershed algorithm
    """
    height, width = image.shape

    distance_image = -image  # Invert so peaks become valleys

    markers = jnp.zeros((height, width), dtype=jnp.int32)

    # Place markers at peak positions
    def place_marker(i, peak_pos):
        y, x = peak_pos.astype(jnp.int32)
        is_valid = (y >= 0) & (y < height) & (x >= 0) & (x < width)

        marker_value = jax.lax.cond(is_valid, lambda: i + 1, lambda: 0)  # Label from 1

        return jax.lax.cond(
            is_valid, lambda: markers.at[y, x].set(marker_value), lambda: markers
        )

    # Sequential marker placement
    for i in range(peaks.shape[0]):
        markers = place_marker(i, peaks[i])

    # Apply watershed algorithm
    watershed_labels = watershed_segmentation(
        distance_image,
        markers,
        max_iterations=max_iterations,
        mask=mask,
    )

    return watershed_labels
