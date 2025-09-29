"""
Compilation utilities for nnU-Net trainers.

This module contains utilities for determining whether torch.compile should be used
based on device type and environment settings.
"""
import os
import torch


def _do_i_compile(device: torch.device, print_func=print):
    """
    Determines whether torch.compile should be used based on device type and environment settings.

    Args:
        device: The torch device being used
        print_func: Function to use for printing messages (default: print)

    Returns:
        bool: True if torch.compile should be used, False otherwise
    """
    # new default: compile is enabled!

    # compile does not work on mps
    if device == torch.device('mps'):
        if 'nnUNet_compile' in os.environ.keys() and os.environ['nnUNet_compile'].lower() in ('true', '1', 't'):
            print_func("INFO: torch.compile disabled because of unsupported mps device")
        return False

    # CPU compile crashes for 2D models. Not sure if we even want to support CPU compile!? Better disable
    if device == torch.device('cpu'):
        if 'nnUNet_compile' in os.environ.keys() and os.environ['nnUNet_compile'].lower() in ('true', '1', 't'):
            print_func("INFO: torch.compile disabled because device is CPU")
        return False

    # default torch.compile doesn't work on windows because there are apparently no triton wheels for it
    # https://discuss.pytorch.org/t/windows-support-timeline-for-torch-compile/182268/2
    if os.name == 'nt':
        if 'nnUNet_compile' in os.environ.keys() and os.environ['nnUNet_compile'].lower() in ('true', '1', 't'):
            print_func("INFO: torch.compile disabled because Windows is not natively supported. If "
                      "you know what you are doing, check https://discuss.pytorch.org/t/windows-support-timeline-for-torch-compile/182268/2")
        return False

    if 'nnUNet_compile' not in os.environ.keys():
        return True
    else:
        return os.environ['nnUNet_compile'].lower() in ('true', '1', 't')