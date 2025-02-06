from importlib import import_module

import torch
import numpy as np

def get_dataset(cfg, mode='train', **kwargs):
    dpath= f"src.data.{cfg.dataset_type}"
    ds = import_module(dpath).CustomDataset(cfg=cfg, mode=mode, **kwargs)
    return ds

def get_dataloader(ds, cfg, mode='train'):
    if mode == 'train':
        dl = torch.utils.data.DataLoader(
            ds,
            sampler= None,
            shuffle= True,
            batch_size= cfg.batch_size,
            num_workers= cfg.num_workers,
            pin_memory= cfg.pin_memory,
            drop_last= cfg.drop_last,
        )
    elif mode in ["val", "test"]:
        sampler = torch.utils.data.SequentialSampler(ds)
        dl = torch.utils.data.DataLoader(
            ds,
            sampler= sampler,
            shuffle= False,
            batch_size= cfg.batch_size_val if cfg.batch_size_val is not None else cfg.batch_size,
            num_workers= cfg.num_workers,
            pin_memory= cfg.pin_memory,
        )
    
    print(f"{mode.upper()}: dataset {len(ds)} dataloader {len(dl)}")
    return dl