from typing import List

import numpy as np


def validate_nibabel_geometry(properties_image: dict, properties_seg: dict,
                             image_files: List[str], label_file: str) -> bool:
    """
    Validate NiBabel-specific geometry properties (affine matrices).

    Args:
        properties_image: Image properties dictionary
        properties_seg: Segmentation properties dictionary
        image_files: List of image file paths (for error reporting)
        label_file: Label file path (for error reporting)

    Returns:
        True if validation passes (warning only for nibabel), False for critical errors
    """
    if 'nibabel_stuff' not in properties_image:
        return True

    # this image was read with NibabelIO
    affine_image = properties_image['nibabel_stuff']['original_affine']
    affine_seg = properties_seg['nibabel_stuff']['original_affine']
    if not np.allclose(affine_image, affine_seg):
        print('WARNING: Affine is not the same for image and seg! \nAffine image: %s \nAffine seg: %s\n'
              'Image files: %s. \nSeg file: %s.\nThis can be a problem but doesn\'t have to be. Please run '
              'nnUNetv2_plot_overlay_pngs to verify if everything is OK!\n'
              % (affine_image, affine_seg, image_files, label_file))
        # This is a warning, not an error
    return True


def validate_sitk_geometry(properties_image: dict, properties_seg: dict,
                          image_files: List[str], label_file: str) -> bool:
    """
    Validate SimpleITK-specific geometry properties (origin and direction).

    Args:
        properties_image: Image properties dictionary
        properties_seg: Segmentation properties dictionary
        image_files: List of image file paths (for error reporting)
        label_file: Label file path (for error reporting)

    Returns:
        True if validation passes (warnings only), False for critical errors
    """
    if 'sitk_stuff' not in properties_image:
        return True

    # this image was read with SimpleITKIO
    # spacing has already been checked, only check direction and origin
    origin_image = properties_image['sitk_stuff']['origin']
    origin_seg = properties_seg['sitk_stuff']['origin']
    if not np.allclose(origin_image, origin_seg):
        print('Warning: Origin mismatch between segmentation and corresponding images. \nOrigin images: %s. '
              '\nOrigin seg: %s. \nImage files: %s. \nSeg file: %s\n' %
              (origin_image, origin_seg, image_files, label_file))

    direction_image = properties_image['sitk_stuff']['direction']
    direction_seg = properties_seg['sitk_stuff']['direction']
    if not np.allclose(direction_image, direction_seg):
        print('Warning: Direction mismatch between segmentation and corresponding images. \nDirection images: %s. '
              '\nDirection seg: %s. \nImage files: %s. \nSeg file: %s\n' %
              (direction_image, direction_seg, image_files, label_file))

    # These are warnings, not errors
    return True


def validate_all_geometry(properties_image: dict, properties_seg: dict,
                         image_files: List[str], label_file: str) -> bool:
    """
    Validate all geometry-related properties.

    Args:
        properties_image: Image properties dictionary
        properties_seg: Segmentation properties dictionary
        image_files: List of image file paths (for error reporting)
        label_file: Label file path (for error reporting)

    Returns:
        True if validation passes, False for critical errors
    """
    ret = True

    # Validate nibabel geometry (warnings only)
    validate_nibabel_geometry(properties_image, properties_seg, image_files, label_file)

    # Validate SimpleITK geometry (warnings only)
    validate_sitk_geometry(properties_image, properties_seg, image_files, label_file)

    return ret