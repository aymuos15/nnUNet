from typing import List, Type

import numpy as np
import pandas as pd
from nnunetv2.imageio.base_reader_writer import BaseReaderWriter


def verify_labels(label_file: str, readerclass: Type[BaseReaderWriter], expected_labels: List[int]) -> bool:
    """
    Verify that a label file contains only the expected labels.

    Args:
        label_file: Path to the label file
        readerclass: Reader class to use for loading
        expected_labels: List of expected label values

    Returns:
        True if labels are valid, False otherwise
    """
    rw = readerclass()
    seg, properties = rw.read_seg(label_file)
    found_labels = np.sort(pd.unique(seg.ravel()))  # np.unique(seg)
    unexpected_labels = [i for i in found_labels if i not in expected_labels]

    if len(found_labels) == 0 and found_labels[0] == 0:
        print('WARNING: File %s only has label 0 (which should be background). This may be intentional or not, '
              'up to you.' % label_file)

    if len(unexpected_labels) > 0:
        print("Error: Unexpected labels found in file %s.\nExpected: %s\nFound: %s" % (label_file, expected_labels,
                                                                                       found_labels))
        return False
    return True


def validate_labels_consecutive(expected_labels: List[int]) -> bool:
    """
    Validate that labels are in consecutive order (0, 1, 2, ...).

    Args:
        expected_labels: List of expected label values

    Returns:
        True if labels are consecutive, False otherwise
    """
    labels_valid_consecutive = np.ediff1d(expected_labels) == 1
    if not all(labels_valid_consecutive):
        invalid_labels = np.array(expected_labels)[1:][~labels_valid_consecutive]
        print(f'Labels must be in consecutive order (0, 1, 2, ...). The labels {invalid_labels} do not satisfy this restriction')
        return False
    return True