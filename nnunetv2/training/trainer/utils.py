"""
Utility functions for trainer operations.

This module provides:
- DDP initialization and management
- Helper functions for training
"""

import numpy as np
import torch
from torch import distributed as dist
from torch.cuda import device_count
import os




# ==== DDP Utilities ====

def initialize_ddp_device(device: torch.device):
    """
    Initialize DDP device settings and determine local rank.

    Args:
        device: Initial torch device

    Returns:
        tuple: (is_ddp, local_rank, final_device)
    """
    is_ddp = dist.is_available() and dist.is_initialized()
    local_rank = 0 if not is_ddp else dist.get_rank()

    # print what device we are using
    if is_ddp:  # implicitly it's clear that we use cuda in this case
        print(f"I am local rank {local_rank}. {device_count()} GPUs are available. The world size is "
              f"{dist.get_world_size()}."
              f"Setting device to {device}")
        final_device = torch.device(type='cuda', index=local_rank)
    else:
        if device.type == 'cuda':
            # we might want to let the user pick this but for now please pick the correct GPU with CUDA_VISIBLE_DEVICES=X
            final_device = torch.device(type='cuda', index=0)
        else:
            final_device = device
        print(f"Using device: {final_device}")

    return is_ddp, local_rank, final_device


def set_batch_size_and_oversample(is_ddp: bool, configuration_batch_size: int, oversample_foreground_percent: float):
    """
    Set batch size and oversample percentage for DDP or single GPU training.

    Args:
        is_ddp: Whether we're using DDP
        configuration_batch_size: Batch size from configuration
        oversample_foreground_percent: Original oversample percentage

    Returns:
        tuple: (batch_size, oversample_foreground_percent)
    """
    if not is_ddp:
        # set batch size to what the plan says, leave oversample untouched
        return configuration_batch_size, oversample_foreground_percent
    else:
        # batch size is distributed over DDP workers and we need to change oversample_percent for each worker

        world_size = dist.get_world_size()
        my_rank = dist.get_rank()

        global_batch_size = configuration_batch_size
        assert global_batch_size >= world_size, 'Cannot run DDP if the batch size is smaller than the number of ' \
                                                'GPUs... Duh.'

        batch_size_per_GPU = [global_batch_size // world_size] * world_size
        batch_size_per_GPU = [batch_size_per_GPU[i] + 1
                              if (batch_size_per_GPU[i] * world_size + i) < global_batch_size
                              else batch_size_per_GPU[i]
                              for i in range(len(batch_size_per_GPU))]
        assert sum(batch_size_per_GPU) == global_batch_size

        sample_id_low = 0 if my_rank == 0 else np.sum(batch_size_per_GPU[:my_rank])
        sample_id_high = np.sum(batch_size_per_GPU[:my_rank + 1])

        # This is how oversampling is determined in DataLoader
        # round(self.batch_size * (1 - self.oversample_foreground_percent))
        # We need to use the same scheme here because an oversample of 0.33 with a batch size of 2 will be rounded
        # to an oversample of 0.5 (1 sample random, one oversampled). This may get lost if we just numerically
        # compute oversample
        oversample = [True if not i < round(global_batch_size * (1 - oversample_foreground_percent)) else False
                      for i in range(global_batch_size)]

        if sample_id_high / global_batch_size < (1 - oversample_foreground_percent):
            oversample_percent = 0.0
        elif sample_id_low / global_batch_size > (1 - oversample_foreground_percent):
            oversample_percent = 1.0
        else:
            oversample_percent = sum(oversample[sample_id_low:sample_id_high]) / batch_size_per_GPU[my_rank]

        print("worker", my_rank, "oversample", oversample_percent)
        print("worker", my_rank, "batch_size", batch_size_per_GPU[my_rank])

        return batch_size_per_GPU[my_rank], oversample_percent


def wrap_network_for_ddp(network, is_ddp: bool, local_rank: int):
    """
    Wrap network for DDP if needed.

    Args:
        network: PyTorch network to wrap
        is_ddp: Whether we're using DDP
        local_rank: Local rank for DDP

    Returns:
        Wrapped network (or original if not DDP)
    """
    if is_ddp:
        network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(network)
        from torch.nn.parallel import DistributedDataParallel as DDP
        network = DDP(network, device_ids=[local_rank])
    return network