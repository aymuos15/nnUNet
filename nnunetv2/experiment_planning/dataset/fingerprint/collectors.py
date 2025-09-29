from typing import List, Tuple

import numpy as np


def collect_foreground_intensities(segmentation: np.ndarray, images: np.ndarray, seed: int = 1234,
                                   num_samples: int = 10000) -> Tuple[List[np.ndarray], List[dict]]:
    """
    Collect foreground intensity samples and statistics from images.

    Args:
        segmentation: Segmentation array with shape (1, x, y, z)
        images: Image array with shape (c, x, y, z)
        seed: Random seed for sampling
        num_samples: Number of samples to collect per channel

    Returns:
        Tuple of (intensity samples per channel, intensity statistics per channel)
    """
    assert images.ndim == 4 and segmentation.ndim == 4
    assert not np.any(np.isnan(segmentation)), "Segmentation contains NaN values. grrrr.... :-("
    assert not np.any(np.isnan(images)), "Images contains NaN values. grrrr.... :-("

    rs = np.random.RandomState(seed)

    intensities_per_channel = []
    intensity_statistics_per_channel = []

    # segmentation is 4d: 1,x,y,z. We need to remove the empty dimension for the following code to work
    foreground_mask = segmentation[0] > 0
    percentiles = np.array((0.5, 50.0, 99.5))

    for i in range(len(images)):
        foreground_pixels = images[i][foreground_mask]
        num_fg = len(foreground_pixels)

        # sample with replacement so that we don't get issues with cases that have less than num_samples
        # foreground_pixels. We could also just sample less in those cases but that would than cause these
        # training cases to be underrepresented
        intensities_per_channel.append(
            rs.choice(foreground_pixels, num_samples, replace=True) if num_fg > 0 else [])

        mean, median, mini, maxi, percentile_99_5, percentile_00_5 = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        if num_fg > 0:
            percentile_00_5, median, percentile_99_5 = np.percentile(foreground_pixels, percentiles)
            mean = np.mean(foreground_pixels)
            mini = np.min(foreground_pixels)
            maxi = np.max(foreground_pixels)

        intensity_statistics_per_channel.append({
            'mean': mean,
            'median': median,
            'min': mini,
            'max': maxi,
            'percentile_99_5': percentile_99_5,
            'percentile_00_5': percentile_00_5,
        })

    return intensities_per_channel, intensity_statistics_per_channel


def calculate_global_intensity_statistics(foreground_intensities_per_channel: np.ndarray,
                                        num_channels: int) -> dict:
    """
    Calculate global intensity statistics from collected samples.

    Args:
        foreground_intensities_per_channel: Array of intensity samples per channel
        num_channels: Number of image channels

    Returns:
        Dictionary of intensity statistics per channel
    """
    intensity_statistics_per_channel = {}
    percentiles = np.array((0.5, 50.0, 99.5))

    for i in range(num_channels):
        percentile_00_5, median, percentile_99_5 = np.percentile(foreground_intensities_per_channel[i], percentiles)
        intensity_statistics_per_channel[i] = {
            'mean': float(np.mean(foreground_intensities_per_channel[i])),
            'median': float(median),
            'std': float(np.std(foreground_intensities_per_channel[i])),
            'min': float(np.min(foreground_intensities_per_channel[i])),
            'max': float(np.max(foreground_intensities_per_channel[i])),
            'percentile_99_5': float(percentile_99_5),
            'percentile_00_5': float(percentile_00_5),
        }

    return intensity_statistics_per_channel