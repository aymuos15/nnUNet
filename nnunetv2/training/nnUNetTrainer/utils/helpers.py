"""
General helper utilities for nnU-Net trainers.

This module contains various utility functions that support training operations,
including logging, debug information collection, and deep supervision scaling.
"""
import os
import sys
import subprocess
from datetime import datetime
from time import time, sleep
from typing import Union, List, Tuple

import numpy as np
import torch
from batchgenerators.utilities.file_and_folder_operations import save_json, join


def save_debug_information(trainer_obj):
    """
    Save debug information about the trainer to debug.json.

    Args:
        trainer_obj: The trainer object to extract debug info from
    """
    if trainer_obj.local_rank == 0:
        dct = {}
        for k in trainer_obj.__dir__():
            if not k.startswith("__"):
                if not callable(getattr(trainer_obj, k)) or k in ['loss', ]:
                    dct[k] = str(getattr(trainer_obj, k))
                elif k in ['network', ]:
                    dct[k] = str(getattr(trainer_obj, k).__class__.__name__)
                else:
                    # print(k)
                    pass
            if k in ['dataloader_train', 'dataloader_val']:
                if hasattr(getattr(trainer_obj, k), 'generator'):
                    dct[k + '.generator'] = str(getattr(trainer_obj, k).generator)
                if hasattr(getattr(trainer_obj, k), 'num_processes'):
                    dct[k + '.num_processes'] = str(getattr(trainer_obj, k).num_processes)
                if hasattr(getattr(trainer_obj, k), 'transform'):
                    dct[k + '.transform'] = str(getattr(trainer_obj, k).transform)

        hostname = subprocess.getoutput(['hostname'])
        dct['hostname'] = hostname
        torch_version = torch.__version__
        if trainer_obj.device.type == 'cuda':
            gpu_name = torch.cuda.get_device_name()
            dct['gpu_name'] = gpu_name
            cudnn_version = torch.backends.cudnn.version()
        else:
            cudnn_version = 'None'
        dct['device'] = str(trainer_obj.device)
        dct['torch_version'] = torch_version
        dct['cudnn_version'] = cudnn_version
        save_json(dct, join(trainer_obj.output_folder, "debug.json"))


def print_to_log_file(log_file: str, local_rank: int, *args,
                      also_print_to_console: bool = True, add_timestamp: bool = True):
    """
    Print messages to log file with optional console output and timestamp.

    Args:
        log_file: Path to the log file
        local_rank: Local rank for distributed training (only rank 0 writes)
        *args: Arguments to print
        also_print_to_console: Whether to also print to console
        add_timestamp: Whether to add timestamp to log entries
    """
    if local_rank == 0:
        timestamp = time()
        dt_object = datetime.fromtimestamp(timestamp)

        if add_timestamp:
            args = (f"{dt_object}:", *args)

        successful = False
        max_attempts = 5
        ctr = 0
        while not successful and ctr < max_attempts:
            try:
                with open(log_file, 'a+') as f:
                    for a in args:
                        f.write(str(a))
                        f.write(" ")
                    f.write("\n")
                successful = True
            except IOError:
                print(f"{datetime.fromtimestamp(timestamp)}: failed to log: ", sys.exc_info())
                sleep(0.5)
                ctr += 1
        if also_print_to_console:
            print(*args)
    elif also_print_to_console:
        print(*args)


def get_deep_supervision_scales(enable_deep_supervision: bool, pool_op_kernel_sizes):
    """
    Calculate deep supervision scales based on pool operation kernel sizes.

    Args:
        enable_deep_supervision: Whether deep supervision is enabled
        pool_op_kernel_sizes: Pool operation kernel sizes from configuration manager

    Returns:
        List of scales for deep supervision or None if disabled
    """
    if enable_deep_supervision:
        deep_supervision_scales = list(list(i) for i in 1 / np.cumprod(np.vstack(
            pool_op_kernel_sizes), axis=0))[:-1]
    else:
        deep_supervision_scales = None  # for train and val_transforms
    return deep_supervision_scales


def build_network_architecture(architecture_class_name: str,
                               arch_init_kwargs: dict,
                               arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                               num_input_channels: int,
                               num_output_channels: int,
                               enable_deep_supervision: bool = True):
    """
    Build network architecture according to the plans.

    This is where you build the architecture according to the plans. There is no obligation to use
    get_network_from_plans, this is just a utility we use for the nnU-Net default architectures. You can do what
    you want. Even ignore the plans and just return something static (as long as it can process the requested
    patch size) but don't bug us with your bugs arising from fiddling with this :-P

    This is the function that is called in inference as well! This is needed so that all network architecture
    variants can be loaded at inference time (inference will use the same nnUNetTrainer that was used for
    training, so if you change the network architecture during training by deriving a new trainer class then
    inference will know about it).

    If you need to know how many segmentation outputs your custom architecture needs to have, use the following snippet:
    > label_manager = plans_manager.get_label_manager(dataset_json)
    > label_manager.num_segmentation_heads
    (why so complicated? -> We can have either classical training (classes) or regions. If we have regions,
    the number of outputs is != the number of classes. Also there is the ignore label for which no output
    should be generated. label_manager takes care of all that for you.)

    Args:
        architecture_class_name: Name of the architecture class
        arch_init_kwargs: Architecture initialization kwargs
        arch_init_kwargs_req_import: Required imports for architecture
        num_input_channels: Number of input channels
        num_output_channels: Number of output channels
        enable_deep_supervision: Whether to enable deep supervision

    Returns:
        torch.nn.Module: The built network
    """
    from nnunetv2.utilities.planning.get_network_from_plans import get_network_from_plans

    return get_network_from_plans(
        architecture_class_name,
        arch_init_kwargs,
        arch_init_kwargs_req_import,
        num_input_channels,
        num_output_channels,
        allow_init=True,
        deep_supervision=enable_deep_supervision)