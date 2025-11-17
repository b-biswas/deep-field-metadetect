import jax.numpy as jnp
import numpy as np

from deep_field_metadetect.jaxify.jax_detection import (
    detect_galaxies,
    local_maxima_filter,
    peak_finder,
    refine_centroid,
)


def create_gaussian_blob(shape, center, sigma=1.0, amplitude=1.0):
    """
    Create a 2D Gaussian blob for testing.

    Parameters:
    -----------
    shape : tuple
        Shape of the output array (height, width)
    center : tuple
        Center position (y, x) of the Gaussian
    sigma : float
        Standard deviation of the Gaussian
    amplitude : float
        Peak amplitude of the Gaussian

    Returns:
    --------
    jnp.ndarray
        2D array containing the Gaussian blob
    """
    y, x = jnp.meshgrid(jnp.arange(shape[0]), jnp.arange(shape[1]), indexing="ij")
    cy, cx = center

    gaussian = amplitude * jnp.exp(-((y - cy) ** 2 + (x - cx) ** 2) / (2 * sigma**2))
    return gaussian


def create_multiple_gaussian_blobs(shape, centers, sigmas=None, amplitudes=None):
    """
    Create multiple Gaussian blobs in a single image.

    Parameters:
    -----------
    shape : tuple
        Shape of the output array (height, width)
    centers : list of tuples
        List of center positions [(y1, x1), (y2, x2), ...]
    sigmas : list of floats or None
        Standard deviations for each blob. If None, uses 1.0 for all
    amplitudes : list of floats or None
        Amplitudes for each blob. If None, uses 1.0 for all

    Returns:
    --------
    jnp.ndarray
        2D array containing all Gaussian blobs
    """
    if sigmas is None:
        sigmas = [1.0] * len(centers)
    if amplitudes is None:
        amplitudes = [1.0] * len(centers)

    image = jnp.zeros(shape)
    for center, sigma, amplitude in zip(centers, sigmas, amplitudes):
        blob = create_gaussian_blob(shape, center, sigma, amplitude)
        image = image + blob

    return image


# -------------------
# Test peak detection
# -------------------


def single_gaussian():
    """Test detection of multiple well-separated Gaussian peaks."""
    centers = [(5, 5)]
    amplitudes = [1.0]
    max_objects = 10
    image = create_multiple_gaussian_blobs(
        (10, 10), centers, sigmas=[1.0], amplitudes=amplitudes
    )

    peak = peak_finder(image, max_objects=max_objects)

    assert len(peak) == max_objects
    assert (peak[0][0] == centers[0][0]) & (peak[0][1] == centers[0][1])


def test_multiple_separated_gaussians():
    """Test detection of multiple well-separated Gaussian peaks."""
    centers = [(2, 2), (2, 7), (7, 2), (7, 7)]
    amplitudes = [1.0, 1.5, 2.0, 0.8]
    image = create_multiple_gaussian_blobs(
        (10, 10), centers, sigmas=[1.0] * 4, amplitudes=amplitudes
    )

    result = local_maxima_filter(image, window_size=3, threshold=0.0)

    # All centers should be detected as peaks
    for center in centers:
        assert result[center[0], center[1]]


def test_threshold_filtering_gaussians():
    """Test that Gaussian peaks below threshold are filtered out."""
    centers = [(3, 3), (3, 9)]
    amplitudes = [0.5, 2.0]  # First below threshold, second above
    image = create_multiple_gaussian_blobs((12, 12), centers, amplitudes=amplitudes)

    result = local_maxima_filter(image, window_size=3, threshold=0.6)

    # Only the high amplitude peak should be detected
    assert not result[3, 3]  # Below threshold
    assert result[3, 9]  # Above threshold


def test_overlapping_gaussians():
    """Test behavior with overlapping Gaussian blobs."""
    # Two Gaussians close together
    centers = [(4, 4), (4, 6)]
    image = create_multiple_gaussian_blobs(
        (9, 9), centers, sigmas=[1.5, 1.5], amplitudes=[1.0, 1.0]
    )

    result = local_maxima_filter(image, window_size=3, threshold=0.0)

    # Depending on overlap, may detect one or both peaks
    # At minimum, should detect at least one peak in the region
    peak_region = result[3:6, 3:7]
    assert jnp.any(peak_region)


def test_edge_case_detection():
    """Test detection of edge cases and boundary conditions."""
    # Single pixel "galaxy"
    image = jnp.zeros((7, 7))
    image = image.at[3, 3].set(5.0)

    peaks, refined, border_flags = detect_galaxies(
        image, threshold=1.0, window_size=3, refine_centroids=True, max_objects=5
    )

    valid_peaks = peaks[peaks[:, 0] > 0]

    assert len(valid_peaks) == 1
    assert jnp.array_equal(valid_peaks[0], jnp.array([3, 3]))


# ------------------------
# Test Centriod Refinement
# ------------------------


def test_gaussian_centroid_refinement():
    """Test centroid refinement on slightly off-center Gaussian."""
    # Create Gaussian slightly off-grid
    true_center = (4.3, 4.7)
    image = create_gaussian_blob((9, 9), true_center, sigma=1.5, amplitude=2.0)

    # Start refinement from nearest grid point
    initial_peak = (4, 5)
    refined_peak, near_border = refine_centroid(image, initial_peak, window_size=5)

    # Refined position should be closer to true center
    initial_distance = np.sqrt(
        (initial_peak[0] - true_center[0]) ** 2
        + (initial_peak[1] - true_center[1]) ** 2
    )
    refined_distance = np.sqrt(
        (refined_peak[0] - true_center[0]) ** 2
        + (refined_peak[1] - true_center[1]) ** 2
    )

    assert refined_distance < initial_distance
    assert not near_border


def test_near_border():
    """Test near border case."""
    # Create two overlapping Gaussians to make asymmetric peak
    centers = [(3, 3)]
    amplitudes = [1.0]
    image = create_multiple_gaussian_blobs(
        (5, 5), centers, sigmas=[1.0], amplitudes=amplitudes
    )

    refined_pos, near_border = refine_centroid(image, (4, 4), window_size=5)

    assert (refined_pos[0] == 4) & (refined_pos[1] == 4)  # refined is same as input
    assert near_border


# -----------------------------
# Test galaxy dection in fields
# -----------------------------


def test_complete_gaussian_detection():
    """Test complete detection pipeline on Gaussian galaxies."""
    centers = [(5, 5), (5, 15), (15, 5), (15, 15)]
    amplitudes = [2.0, 1.5, 1.8, 1.2]
    sigmas = [1.5, 1.2, 1.3, 1]

    image = create_multiple_gaussian_blobs(
        (21, 21), centers, sigmas=sigmas, amplitudes=amplitudes
    )

    peaks, refined, _ = detect_galaxies(
        image, threshold=0.5, window_size=5, refine_centroids=True, max_objects=10
    )

    valid_peaks = peaks[peaks[:, 0] > 0]
    valid_refined = refined[peaks[:, 0] > 0]

    # Should detect all 4 galaxies
    assert len(valid_peaks) == 4

    assert np.all(valid_peaks == jnp.array(centers))

    # Refinement should improve positions for off-grid centers
    for i in range(len(valid_refined)):
        # Refined positions should be reasonable
        assert np.abs(np.asarray(centers)[i, 0] - valid_refined[i, 0]) < 0.5
        assert np.abs(np.asarray(centers)[i, 1] - valid_refined[i, 1]) < 0.5


def test_detection_with_noise():
    """Test detection robustness with added noise."""
    np.random.seed(42)
    # Create clean Gaussian
    peak_location = (6, 6)
    image_clean = create_gaussian_blob((15, 15), (6, 6), sigma=1.0, amplitude=2.0)

    # Add noise
    noise = jnp.array(np.random.normal(0, 0.2, image_clean.shape))
    image_noisy = image_clean + noise

    _, refined, _ = detect_galaxies(
        image_noisy, threshold=1.0, window_size=5, refine_centroids=True, max_objects=5
    )

    valid_peaks = refined[refined[:, 0] > 0]

    # Should still detect the main peak despite noise
    assert len(valid_peaks) >= 1

    # Main peak should be near expected position
    main_peak = valid_peaks[0]
    distance_to_true = np.sqrt(
        (main_peak[0] - peak_location[0]) ** 2 + (main_peak[1] - peak_location[1]) ** 2
    )
    assert distance_to_true < 0.5


def test_faint_galaxy_detection():
    """Test detection of faint galaxies."""
    centers = [(8, 6), (8, 12)]
    amplitudes = [2.0, 0.8]  # One bright, one faint

    image = create_multiple_gaussian_blobs(
        (17, 17), centers, sigmas=[1.5, 1.5], amplitudes=amplitudes
    )

    # Test with threshold that should catch both
    peaks_low, _, _ = detect_galaxies(image, threshold=0.3, max_objects=5)
    valid_low = peaks_low[peaks_low[:, 0] > 0]

    # Test with threshold that should only catch bright one
    peaks_high, _, _ = detect_galaxies(image, threshold=1.0, max_objects=5)
    valid_high = peaks_high[peaks_high[:, 0] > 0]

    assert len(valid_low) == 2
    assert len(valid_high) == 1
