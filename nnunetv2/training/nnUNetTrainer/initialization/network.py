import os
from typing import Union, List, Tuple
import torch
from torch import nn
from torch._dynamo import OptimizedModule
from batchgenerators.utilities.file_and_folder_operations import join

from nnunetv2.utilities.planning.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.core.helpers import empty_cache


def build_network_architecture(architecture_class_name: str,
                               arch_init_kwargs: dict,
                               arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                               num_input_channels: int,
                               num_output_channels: int,
                               enable_deep_supervision: bool = True) -> nn.Module:
    """
    This is where you build the architecture according to the plans. There is no obligation to use
    get_network_from_plans, this is just a utility we use for the nnU-Net default architectures. You can do what
    you want. Even ignore the plans and just return something static (as long as it can process the requested
    patch size)
    but don't bug us with your bugs arising from fiddling with this :-P
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

    """
    return get_network_from_plans(
        architecture_class_name,
        arch_init_kwargs,
        arch_init_kwargs_req_import,
        num_input_channels,
        num_output_channels,
        allow_init=True,
        deep_supervision=enable_deep_supervision)


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