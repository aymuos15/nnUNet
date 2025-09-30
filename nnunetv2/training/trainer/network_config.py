"""
Network configuration utilities for training.

This module handles training-specific network configuration:
- Torch compilation decisions
- Deep supervision control
- Network architecture visualization
"""

import os
from torch import nn
from torch._dynamo import OptimizedModule
from batchgenerators.utilities.file_and_folder_operations import join
from nnunetv2.utilities.core.helpers import empty_cache
import torch


def _do_i_compile(trainer_instance):
    """
    Determine whether to compile the network using torch.compile.

    Args:
        trainer_instance: The nnUNetTrainer instance

    Returns:
        bool: Whether to compile the network
    """
    # new default: compile is enabled!

    # compile does not work on mps
    if trainer_instance.device == torch.device('mps'):
        if 'nnUNet_compile' in os.environ.keys() and os.environ['nnUNet_compile'].lower() in ('true', '1', 't'):
            trainer_instance.print_to_log_file("INFO: torch.compile disabled because of unsupported mps device")
        return False

    # CPU compile crashes for 2D models. Not sure if we even want to support CPU compile!? Better disable
    if trainer_instance.device == torch.device('cpu'):
        if 'nnUNet_compile' in os.environ.keys() and os.environ['nnUNet_compile'].lower() in ('true', '1', 't'):
            trainer_instance.print_to_log_file("INFO: torch.compile disabled because device is CPU")
        return False

    # default torch.compile doesn't work on windows because there are apparently no triton wheels for it
    # https://discuss.pytorch.org/t/windows-support-timeline-for-torch-compile/182268/2
    if os.name == 'nt':
        if 'nnUNet_compile' in os.environ.keys() and os.environ['nnUNet_compile'].lower() in ('true', '1', 't'):
            trainer_instance.print_to_log_file("INFO: torch.compile disabled because Windows is not natively supported. If "
                                   "you know what you are doing, check https://discuss.pytorch.org/t/windows-support-timeline-for-torch-compile/182268/2")
        return False

    if 'nnUNet_compile' not in os.environ.keys():
        return True
    else:
        return os.environ['nnUNet_compile'].lower() in ('true', '1', 't')


def plot_network_architecture(trainer_instance):
    """
    Plot the network architecture and save it as a PDF.

    Args:
        trainer_instance: The nnUNetTrainer instance
    """
    if _do_i_compile(trainer_instance):
        trainer_instance.print_to_log_file("Unable to plot network architecture: nnUNet_compile is enabled!")
        return

    if trainer_instance.local_rank == 0:
        try:
            # raise NotImplementedError('hiddenlayer no longer works and we do not have a viable alternative :-(')
            # pip install git+https://github.com/saugatkandel/hiddenlayer.git

            # from torchviz import make_dot
            # # not viable.
            # make_dot(tuple(self.network(torch.rand((1, self.num_input_channels,
            #                                         *self.configuration_manager.patch_size),
            #                                        device=self.device)))).render(
            #     join(self.output_folder, "network_architecture.pdf"), format='pdf')
            # self.optimizer.zero_grad()

            # broken.

            import hiddenlayer as hl
            g = hl.build_graph(trainer_instance.network,
                               torch.rand((1, trainer_instance.num_input_channels,
                                           *trainer_instance.configuration_manager.patch_size),
                                          device=trainer_instance.device),
                               transforms=None)
            g.save(join(trainer_instance.output_folder, "network_architecture.pdf"))
            del g
        except Exception as e:
            trainer_instance.print_to_log_file("Unable to plot network architecture:")
            trainer_instance.print_to_log_file(e)

            # self.print_to_log_file("\nprinting the network instead:\n")
            # self.print_to_log_file(self.network)
            # self.print_to_log_file("\n")
        finally:
            empty_cache(trainer_instance.device)


def set_deep_supervision_enabled(trainer_instance, enabled: bool):
    """
    Enable or disable deep supervision in the network.
    This function is specific for the default architecture in nnU-Net. If you change the architecture, there are
    chances you need to change this as well!

    Args:
        trainer_instance: The nnUNetTrainer instance
        enabled: Whether to enable deep supervision
    """
    if trainer_instance.is_ddp:
        mod = trainer_instance.network.module
    else:
        mod = trainer_instance.network
    if isinstance(mod, OptimizedModule):
        mod = mod._orig_mod

    mod.decoder.deep_supervision = enabled
