"""
Utilities for determining target spacing and transpose operations for experiment planning.
"""
import numpy as np
from typing import Tuple, List, Union


def determine_fullres_target_spacing(
    dataset_fingerprint: dict,
    anisotropy_threshold: float,
    overwrite_target_spacing: Union[List[float], Tuple[float, ...]] = None
) -> np.ndarray:
    """
    Per default we use the 50th percentile=median for the target spacing. Higher spacing results in smaller data
    and thus faster and easier training. Smaller spacing results in larger data and thus longer and harder training

    For some datasets the median is not a good choice. Those are the datasets where the spacing is very anisotropic
    (for example ACDC with (10, 1.5, 1.5)). These datasets still have examples with a spacing of 5 or 6 mm in the low
    resolution axis. Choosing the median here will result in bad interpolation artifacts that can substantially
    impact performance (due to the low number of slices).

    Args:
        dataset_fingerprint: Dictionary containing dataset statistics including 'spacings' and 'shapes_after_crop'
        anisotropy_threshold: Threshold for detecting anisotropic spacing
        overwrite_target_spacing: Optional override for target spacing

    Returns:
        Target spacing as numpy array
    """
    if overwrite_target_spacing is not None:
        return np.array(overwrite_target_spacing)

    spacings = np.vstack(dataset_fingerprint['spacings'])
    sizes = dataset_fingerprint['shapes_after_crop']

    target = np.percentile(spacings, 50, 0)

    # todo sizes_after_resampling = [compute_new_shape(j, i, target) for i, j in zip(spacings, sizes)]

    target_size = np.percentile(np.vstack(sizes), 50, 0)
    # we need to identify datasets for which a different target spacing could be beneficial. These datasets have
    # the following properties:
    # - one axis which much lower resolution than the others
    # - the lowres axis has much less voxels than the others
    # - (the size in mm of the lowres axis is also reduced)
    worst_spacing_axis = np.argmax(target)
    other_axes = [i for i in range(len(target)) if i != worst_spacing_axis]
    other_spacings = [target[i] for i in other_axes]
    other_sizes = [target_size[i] for i in other_axes]

    has_aniso_spacing = target[worst_spacing_axis] > (anisotropy_threshold * max(other_spacings))
    has_aniso_voxels = target_size[worst_spacing_axis] * anisotropy_threshold < min(other_sizes)

    if has_aniso_spacing and has_aniso_voxels:
        spacings_of_that_axis = spacings[:, worst_spacing_axis]
        target_spacing_of_that_axis = np.percentile(spacings_of_that_axis, 10)
        # don't let the spacing of that axis get higher than the other axes
        if target_spacing_of_that_axis < max(other_spacings):
            target_spacing_of_that_axis = max(max(other_spacings), target_spacing_of_that_axis) + 1e-5
        target[worst_spacing_axis] = target_spacing_of_that_axis
    return target


def determine_transpose(
    dataset_fingerprint: dict,
    anisotropy_threshold: float,
    suppress_transpose: bool = False,
    overwrite_target_spacing: Union[List[float], Tuple[float, ...]] = None
) -> Tuple[List[int], List[int]]:
    """
    Determines transpose operations for the dataset.

    Args:
        dataset_fingerprint: Dictionary containing dataset statistics
        anisotropy_threshold: Threshold for detecting anisotropic spacing
        suppress_transpose: If True, returns identity transpose
        overwrite_target_spacing: Optional override for target spacing

    Returns:
        Tuple of (transpose_forward, transpose_backward)
    """
    if suppress_transpose:
        return [0, 1, 2], [0, 1, 2]

    # todo we should use shapes for that as well. Not quite sure how yet
    target_spacing = determine_fullres_target_spacing(
        dataset_fingerprint, anisotropy_threshold, overwrite_target_spacing
    )

    max_spacing_axis = np.argmax(target_spacing)
    remaining_axes = [i for i in list(range(3)) if i != max_spacing_axis]
    transpose_forward = [max_spacing_axis] + remaining_axes
    transpose_backward = [np.argwhere(np.array(transpose_forward) == i)[0][0] for i in range(3)]
    return transpose_forward, transpose_backward
