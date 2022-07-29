import torch
from torch.utils.data import DistributedSampler

from .dataset import HDMapNetSemanticDataset


def compile_train_data(data_conf):
    return HDMapNetSemanticDataset(data_conf, is_train=True)


def compile_val_data(data_conf):
    return HDMapNetSemanticDataset(data_conf, is_train=False)


def compile_data(data_conf):
    dataset_train = HDMapNetSemanticDataset(data_conf, is_train=True)
    dataset_val = HDMapNetSemanticDataset(data_conf, is_train=False)

    return dataset_train, dataset_val


def compile_dataloader(data_conf, dataset_train, dataset_val):
    sampler_train = DistributedSampler(dataset_train)
    sampler_val = DistributedSampler(dataset_val, shuffle=False)
    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, data_conf.batch_size, drop_last=True
    )
    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_sampler=batch_sampler_train,
        num_workers=data_conf.num_workers,
        pin_memory=data_conf.pin_memory
    )
    val_loader = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=data_conf.eval_batch_size,
        sampler=sampler_val,
        drop_last=False,
        num_workers=data_conf.num_workers,
        pin_memory=data_conf.pin_memory
    )
    return sampler_train, train_loader, val_loader
