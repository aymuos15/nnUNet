from typing import List, Type, Tuple

import numpy as np
from nnunetv2.imageio.base_reader_writer import BaseReaderWriter
from nnunetv2.preprocessing.cropping.cropping import crop_to_nonzero
from .collectors import collect_foreground_intensities


def analyze_case(image_files: List[str], segmentation_file: str, reader_writer_class: Type[BaseReaderWriter],
                 num_samples: int = 10000) -> Tuple[Tuple[int, ...], Tuple[float, ...], List[np.ndarray], List[dict], float]:
    """
    Analyze a single training case to extract shape, spacing, and intensity information.

    Args:
        image_files: List of image file paths for this case
        segmentation_file: Path to segmentation file
        reader_writer_class: Reader/writer class for loading files
        num_samples: Number of foreground intensity samples to collect

    Returns:
        Tuple of (shape_after_crop, spacing, foreground_intensities_per_channel,
                 foreground_intensity_stats_per_channel, relative_size_after_cropping)
    """
    rw = reader_writer_class()
    images, properties_images = rw.read_images(image_files)
    segmentation, properties_seg = rw.read_seg(segmentation_file)

    # we no longer crop and save the cropped images before this is run. Instead we run the cropping on the fly.
    # Downside is that we need to do this twice (once here and once during preprocessing). Upside is that we don't
    # need to save the cropped data anymore. Given that cropping is not too expensive it makes sense to do it this
    # way. This is only possible because we are now using our new input/output interface.
    data_cropped, seg_cropped, bbox = crop_to_nonzero(images, segmentation)

    foreground_intensities_per_channel, foreground_intensity_stats_per_channel = \
        collect_foreground_intensities(seg_cropped, data_cropped, num_samples=num_samples)

    spacing = properties_images['spacing']

    shape_before_crop = images.shape[1:]
    shape_after_crop = data_cropped.shape[1:]
    relative_size_after_cropping = np.prod(shape_after_crop) / np.prod(shape_before_crop)

    return shape_after_crop, spacing, foreground_intensities_per_channel, foreground_intensity_stats_per_channel, \
           relative_size_after_cropping


def aggregate_case_results(results: List[Tuple]) -> Tuple[List, List, np.ndarray, float]:
    """
    Aggregate results from multiple case analyses.

    Args:
        results: List of results from analyze_case calls

    Returns:
        Tuple of (shapes_after_crop, spacings, foreground_intensities_per_channel, median_relative_size_after_cropping)
    """
    shapes_after_crop = [r[0] for r in results]
    spacings = [r[1] for r in results]

    # Concatenate foreground intensities across all cases
    num_channels = len(results[0][2])
    foreground_intensities_per_channel = [
        np.concatenate([r[2][i] for r in results]) for i in range(num_channels)
    ]
    foreground_intensities_per_channel = np.array(foreground_intensities_per_channel)

    median_relative_size_after_cropping = np.median([r[4] for r in results], 0)

    return shapes_after_crop, spacings, foreground_intensities_per_channel, median_relative_size_after_cropping


def create_fingerprint_dict(spacings: List, shapes_after_crop: List,
                           intensity_statistics_per_channel: dict,
                           median_relative_size_after_cropping: float) -> dict:
    """
    Create the final fingerprint dictionary.

    Args:
        spacings: List of spacings from all cases
        shapes_after_crop: List of shapes after cropping from all cases
        intensity_statistics_per_channel: Global intensity statistics
        median_relative_size_after_cropping: Median relative size after cropping

    Returns:
        Complete fingerprint dictionary
    """
    return {
        "spacings": spacings,
        "shapes_after_crop": shapes_after_crop,
        'foreground_intensity_properties_per_channel': intensity_statistics_per_channel,
        "median_relative_size_after_cropping": median_relative_size_after_cropping
    }