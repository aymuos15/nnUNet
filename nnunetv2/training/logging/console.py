"""
Console and file logging utilities for training.

This module handles:
- Logging messages to file and console
- Printing training plans and configuration
- Saving debug information
"""

import os
import sys
import subprocess
from copy import deepcopy
from datetime import datetime
from time import time, sleep

import torch
from batchgenerators.utilities.file_and_folder_operations import join, save_json


def print_to_log_file(trainer_instance, *args, also_print_to_console=True, add_timestamp=True):
    """Wrapper function for print_to_log_file."""
    logger = create_logging_instance(
        trainer_instance.log_file,
        trainer_instance.output_folder,
        trainer_instance.local_rank,
        trainer_instance.plans_manager,
        trainer_instance.configuration_manager,
        trainer_instance.configuration_name,
        trainer_instance.device
    )
    return logger.print_to_log_file(*args, also_print_to_console=also_print_to_console, add_timestamp=add_timestamp)


def print_plans(trainer_instance):
    """Wrapper function for print_plans."""
    logger = create_logging_instance(
        trainer_instance.log_file,
        trainer_instance.output_folder,
        trainer_instance.local_rank,
        trainer_instance.plans_manager,
        trainer_instance.configuration_manager,
        trainer_instance.configuration_name,
        trainer_instance.device
    )
    return logger.print_plans()


def _save_debug_information(trainer_instance):
    """Wrapper function for _save_debug_information."""
    logger = create_logging_instance(
        trainer_instance.log_file,
        trainer_instance.output_folder,
        trainer_instance.local_rank,
        trainer_instance.plans_manager,
        trainer_instance.configuration_manager,
        trainer_instance.configuration_name,
        trainer_instance.device
    )
    return logger._save_debug_information(trainer_instance)


class nnUNetTrainerLogging:
    """
    Extracted logging components from nnUNetTrainer class.
    Contains methods for logging training information, debug data, and plans.
    """

    def __init__(self, log_file, output_folder, local_rank=0, plans_manager=None,
                 configuration_manager=None, configuration_name=None, device=None):
        """
        Initialize the logging component.

        Args:
            log_file: Path to the log file
            output_folder: Directory for output files
            local_rank: Local rank for distributed training (default: 0)
            plans_manager: Plans manager instance
            configuration_manager: Configuration manager instance
            configuration_name: Name of the configuration
            device: Training device
        """
        self.log_file = log_file
        self.output_folder = output_folder
        self.local_rank = local_rank
        self.plans_manager = plans_manager
        self.configuration_manager = configuration_manager
        self.configuration_name = configuration_name
        self.device = device

    def print_to_log_file(self, *args, also_print_to_console=True, add_timestamp=True):
        """
        Print messages to log file and optionally to console.

        Args:
            *args: Arguments to print
            also_print_to_console: Whether to also print to console (default: True)
            add_timestamp: Whether to add timestamp to log entries (default: True)
        """
        if self.local_rank == 0:
            timestamp = time()
            dt_object = datetime.fromtimestamp(timestamp)

            if add_timestamp:
                args = (f"{dt_object}:", *args)

            successful = False
            max_attempts = 5
            ctr = 0
            while not successful and ctr < max_attempts:
                try:
                    with open(self.log_file, 'a+') as f:
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

    def print_plans(self):
        """
        Print training plans and configuration information to log file.
        Requires plans_manager and configuration_manager to be set.
        """
        if self.local_rank == 0:
            if self.plans_manager is None or self.configuration_manager is None:
                self.print_to_log_file("Cannot print plans: plans_manager or configuration_manager not set")
                return

            dct = deepcopy(self.plans_manager.plans)
            del dct['configurations']
            self.print_to_log_file(f"\nThis is the configuration used by this "
                                   f"training:\nConfiguration name: {self.configuration_name}\n",
                                   self.configuration_manager, '\n', add_timestamp=False)
            self.print_to_log_file('These are the global plan.json settings:\n', dct, '\n', add_timestamp=False)

    def _save_debug_information(self, trainer_instance=None):
        """
        Save debug information to a JSON file.

        Args:
            trainer_instance: The trainer instance to extract debug info from.
                            If None, saves basic system information only.
        """
        if self.local_rank == 0:
            dct = {}

            # If trainer instance is provided, extract its attributes
            if trainer_instance is not None:
                for k in trainer_instance.__dir__():
                    if not k.startswith("__"):
                        if not callable(getattr(trainer_instance, k)) or k in ['loss', ]:
                            dct[k] = str(getattr(trainer_instance, k))
                        elif k in ['network', ]:
                            dct[k] = str(getattr(trainer_instance, k).__class__.__name__)
                        else:
                            # Skip other callable attributes
                            pass
                    if k in ['dataloader_train', 'dataloader_val']:
                        dataloader = getattr(trainer_instance, k)
                        if hasattr(dataloader, 'generator'):
                            dct[k + '.generator'] = str(dataloader.generator)
                        if hasattr(dataloader, 'num_processes'):
                            dct[k + '.num_processes'] = str(dataloader.num_processes)
                        if hasattr(dataloader, 'transform'):
                            dct[k + '.transform'] = str(dataloader.transform)

            # System information
            hostname = subprocess.getoutput(['hostname'])
            dct['hostname'] = hostname
            torch_version = torch.__version__

            if self.device is not None and self.device.type == 'cuda':
                gpu_name = torch.cuda.get_device_name()
                dct['gpu_name'] = gpu_name
                cudnn_version = torch.backends.cudnn.version()
            else:
                cudnn_version = 'None'

            dct['device'] = str(self.device) if self.device is not None else 'None'
            dct['torch_version'] = torch_version
            dct['cudnn_version'] = cudnn_version

            save_json(dct, join(self.output_folder, "debug.json"))


def create_logging_instance(log_file, output_folder, local_rank=0, plans_manager=None,
                          configuration_manager=None, configuration_name=None, device=None):
    """
    Factory function to create a logging instance.

    Args:
        log_file: Path to the log file
        output_folder: Directory for output files
        local_rank: Local rank for distributed training (default: 0)
        plans_manager: Plans manager instance
        configuration_manager: Configuration manager instance
        configuration_name: Name of the configuration
        device: Training device

    Returns:
        nnUNetTrainerLogging instance
    """
    return nnUNetTrainerLogging(
        log_file=log_file,
        output_folder=output_folder,
        local_rank=local_rank,
        plans_manager=plans_manager,
        configuration_manager=configuration_manager,
        configuration_name=configuration_name,
        device=device
    )