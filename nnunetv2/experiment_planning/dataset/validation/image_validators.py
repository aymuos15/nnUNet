from typing import List, Type

import numpy as np
from nnunetv2.imageio.base_reader_writer import BaseReaderWriter


def check_cases(image_files: List[str], label_file: str, expected_num_channels: int,
                readerclass: Type[BaseReaderWriter]) -> bool:
    """
    Validate image and segmentation pair for consistency.

    Args:
        image_files: List of image file paths
        label_file: Path to corresponding label file
        expected_num_channels: Expected number of image channels
        readerclass: Reader class to use for loading

    Returns:
        True if validation passes, False otherwise
    """
    rw = readerclass()
    ret = True

    images, properties_image = rw.read_images(image_files)
    segmentation, properties_seg = rw.read_seg(label_file)

    # check for nans
    if np.any(np.isnan(images)):
        print(f'Images contain NaN pixel values. You need to fix that by '
              f'replacing NaN values with something that makes sense for your images!\nImages:\n{image_files}')
        ret = False
    if np.any(np.isnan(segmentation)):
        print(f'Segmentation contains NaN pixel values. You need to fix that.\nSegmentation:\n{label_file}')
        ret = False

    # check shapes
    shape_image = images.shape[1:]
    shape_seg = segmentation.shape[1:]
    if shape_image != shape_seg:
        print('Error: Shape mismatch between segmentation and corresponding images. \nShape images: %s. '
              '\nShape seg: %s. \nImage files: %s. \nSeg file: %s\n' %
              (shape_image, shape_seg, image_files, label_file))
        ret = False

    # check spacings
    spacing_images = properties_image['spacing']
    spacing_seg = properties_seg['spacing']
    if not np.allclose(spacing_seg, spacing_images):
        print('Error: Spacing mismatch between segmentation and corresponding images. \nSpacing images: %s. '
              '\nSpacing seg: %s. \nImage files: %s. \nSeg file: %s\n' %
              (spacing_images, spacing_seg, image_files, label_file))
        ret = False

    # check modalities
    if not len(images) == expected_num_channels:
        print('Error: Unexpected number of modalities. \nExpected: %d. \nGot: %d. \nImages: %s\n'
              % (expected_num_channels, len(images), image_files))
        ret = False

    return ret


def validate_nan_values(images: np.ndarray, segmentation: np.ndarray,
                        image_files: List[str], label_file: str) -> bool:
    """
    Validate that images and segmentation don't contain NaN values.

    Args:
        images: Image data array
        segmentation: Segmentation data array
        image_files: List of image file paths (for error reporting)
        label_file: Label file path (for error reporting)

    Returns:
        True if no NaN values found, False otherwise
    """
    ret = True
    if np.any(np.isnan(images)):
        print(f'Images contain NaN pixel values. You need to fix that by '
              f'replacing NaN values with something that makes sense for your images!\nImages:\n{image_files}')
        ret = False
    if np.any(np.isnan(segmentation)):
        print(f'Segmentation contains NaN pixel values. You need to fix that.\nSegmentation:\n{label_file}')
        ret = False
    return ret


def validate_shapes_match(images: np.ndarray, segmentation: np.ndarray,
                         image_files: List[str], label_file: str) -> bool:
    """
    Validate that image and segmentation shapes match.

    Args:
        images: Image data array
        segmentation: Segmentation data array
        image_files: List of image file paths (for error reporting)
        label_file: Label file path (for error reporting)

    Returns:
        True if shapes match, False otherwise
    """
    shape_image = images.shape[1:]
    shape_seg = segmentation.shape[1:]
    if shape_image != shape_seg:
        print('Error: Shape mismatch between segmentation and corresponding images. \nShape images: %s. '
              '\nShape seg: %s. \nImage files: %s. \nSeg file: %s\n' %
              (shape_image, shape_seg, image_files, label_file))
        return False
    return True


def validate_spacings_match(properties_image: dict, properties_seg: dict,
                           image_files: List[str], label_file: str) -> bool:
    """
    Validate that image and segmentation spacings match.

    Args:
        properties_image: Image properties dictionary
        properties_seg: Segmentation properties dictionary
        image_files: List of image file paths (for error reporting)
        label_file: Label file path (for error reporting)

    Returns:
        True if spacings match, False otherwise
    """
    spacing_images = properties_image['spacing']
    spacing_seg = properties_seg['spacing']
    if not np.allclose(spacing_seg, spacing_images):
        print('Error: Spacing mismatch between segmentation and corresponding images. \nSpacing images: %s. '
              '\nSpacing seg: %s. \nImage files: %s. \nSeg file: %s\n' %
              (spacing_images, spacing_seg, image_files, label_file))
        return False
    return True


def validate_modality_count(images: np.ndarray, expected_num_channels: int,
                           image_files: List[str]) -> bool:
    """
    Validate that the number of image modalities matches expected count.

    Args:
        images: Image data array
        expected_num_channels: Expected number of channels
        image_files: List of image file paths (for error reporting)

    Returns:
        True if modality count matches, False otherwise
    """
    if not len(images) == expected_num_channels:
        print('Error: Unexpected number of modalities. \nExpected: %d. \nGot: %d. \nImages: %s\n'
              % (expected_num_channels, len(images), image_files))
        return False
    return True