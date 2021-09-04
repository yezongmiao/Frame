import os
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as tmp
from functools import partial
from torch.utils.data.sampler import Sampler
from torch.utils.data import DataLoader as DataLoader


def set_random_seed(seed):
    """Set random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_dist(local_rank=0, backend='nccl'):
    tmp.set_start_method('spawn')
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=backend)


def get_dist_info():
    if dist.is_available():
        initialized = dist.is_initialized()
    else:
        initialized = False

    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    
    return rank, world_size


# ==========
# Dataloader
# ==========


class DistSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.
    
    Modified from torch.utils.data.distributed.DistributedSampler
    Support enlarging the dataset for iteration-based training, for saving
    time when restart the dataloader after each epoch.
    
    Args:
        dataset (torch.utils.data.Dataset): Dataset used for sampling.
        num_replicas (int | None): Number of processes participating in
            the training. It is usually the world_size.
        rank (int | None): Rank of the current process within num_replicas.
        ratio (int): Enlarging ratio. Default: 1.
    """

    def __init__(
            self, dataset, num_replicas=None, rank=None, ratio=1
            ):
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        # enlarged by ratio, and then divided by num_replicas
        self.num_samples = math.ceil(
            len(self.dataset) * ratio / self.num_replicas
            )
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        # deterministically shuffle based on ite_epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        indices = torch.randperm(self.total_size, generator=g).tolist()

        # enlarge indices
        dataset_size = len(self.dataset)
        indices = [v % dataset_size for v in indices]

        # ==========subsample
        # e.g., self.rank=1, self.total_size=4, self.num_replicas=2
        # indices = indices[1:4:2] = indices[i for i in [1, 3]]
        # for the other worker, indices = indices[i for i in [0, 2]]
        # ==========
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        """For shuffling data at each epoch. See train.py."""
        self.epoch = epoch


def create_dataloader(
        dataset, opts_dict, sampler=None, phase='train', seed=None
        ):
    """Create dataloader."""
    if phase == 'train':
        # >I don't know why BasicSR have to detect `is_dist`
        dataloader_args = dict(
            dataset=dataset,
            batch_size=opts_dict['dataset']['train']['batch_size_per_gpu'],
            shuffle=False,  # sampler will shuffle at train.py
            num_workers=opts_dict['dataset']['train']['num_worker_per_gpu'],
            sampler=sampler,
            drop_last=True,
            pin_memory=True
            )
        if sampler is None:
            dataloader_args['shuffle'] = True
        dataloader_args['worker_init_fn'] = partial(
            _worker_init_fn, 
            num_workers=opts_dict['dataset']['train']['num_worker_per_gpu'], 
            rank=opts_dict['train']['rank'],
            seed=seed
            )
        
    elif phase == 'val':
        dataloader_args = dict(
            dataset=dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=False
            )
    
    return DataLoader(**dataloader_args)


def _worker_init_fn(worker_id, num_workers, rank, seed):
    # func for torch.utils.data.DataLoader
    # set the worker seed to num_workers * rank + worker_id + seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ==========
# Loss & Metrics
# ==========

class CharbonnierLoss(torch.nn.Module):
    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss


class PSNR(torch.nn.Module):
    def __init__(self, eps=1e-6):
        super(PSNR, self).__init__()
        self.mse_func = nn.MSELoss()

    def forward(self, X, Y):
        mse = self.mse_func(X, Y)
        psnr = 10 * math.log10(1 / mse.item())
        return psnr


# ==========
# Scheduler
# ==========


import math
from collections import Counter
from torch.optim.lr_scheduler import _LRScheduler




"""
step_size(int)- 学习率下降间隔数，若为 30，则会在 30、 60、 90…个 step 时，将学习率调整为 lr*gamma。
gamma(float)- 学习率调整倍数，默认为 0.1 倍，即下降 10 倍。
last_epoch(int)- 上一个 epoch 数，这个变量用来指示学习率是否需要调整。当last_epoch 符合设定的间隔时，就会对学习率进行调整。

"""




def get_position_from_periods(iteration, cumulative_period):
    """Get the position from a period list.

    It will return the index of the right-closest number in the period list.
    For example, the cumulative_period = [100, 200, 300, 400],
    if iteration == 50, return 0;
    if iteration == 210, return 2;
    if iteration == 300, return 2.

    Args:
        iteration (int): Current iteration.
        cumulative_period (list[int]): Cumulative period list.

    Returns:
        int: The position of the right-closest number in the period list.
    """
    for i, period in enumerate(cumulative_period):
        if iteration <= period:
            return i


class CosineAnnealingRestartLR(_LRScheduler):
    """ Cosine annealing with restarts learning rate scheme.

    An example of config:
    periods = [10, 10, 10, 10]
    restart_weights = [1, 0.5, 0.5, 0.5]
    eta_min=1e-7

    It has four cycles, each has 10 iterations. At 10th, 20th, 30th, the
    scheduler will restart with the weights in restart_weights.

    Args:
        optimizer (torch.nn.optimizer): Torch optimizer.
        periods (list): Period for each cosine anneling cycle.
        restart_weights (list): Restart weights at each restart iteration.
            Default: [1].
        eta_min (float): The mimimum lr. Default: 0.
        last_epoch (int): Used in _LRScheduler. Default: -1.
    """

    def __init__(self,
                 optimizer,
                 periods,
                 restart_weights=[1],
                 eta_min=0,
                 last_epoch=-1):
        self.periods = periods
        self.restart_weights = restart_weights
        self.eta_min = eta_min
        assert (len(self.periods) == len(self.restart_weights)
                ), 'periods and restart_weights should have the same length.'
        self.cumulative_period = [
            sum(self.periods[0:i + 1]) for i in range(0, len(self.periods))
        ]
        super(CosineAnnealingRestartLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        idx = get_position_from_periods(self.last_epoch,
                                        self.cumulative_period)
        current_weight = self.restart_weights[idx]
        nearest_restart = 0 if idx == 0 else self.cumulative_period[idx - 1]
        current_period = self.periods[idx]

        return [
            self.eta_min + current_weight * 0.5 * (base_lr - self.eta_min) *
            (1 + math.cos(math.pi * (
                (self.last_epoch - nearest_restart) / current_period)))
            for base_lr in self.base_lrs
        ]
