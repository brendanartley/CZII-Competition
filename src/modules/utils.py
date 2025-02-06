import os
import time
import pickle
from importlib import import_module

import torch
import torch.nn as nn

def batch_to_device(batch, device, skip_keys=[]):
    batch_dict = {key: batch[key].to(device) for key in batch if key not in skip_keys}
    return batch_dict

def calc_grad_norm(parameters,norm_type=2.):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    if torch.logical_or(total_norm.isnan(), total_norm.isinf()):
        total_norm = None
        
    return total_norm

def get_optimizer(model, cfg):
    optim_modules= import_module("torch.optim")
    c= getattr(optim_modules, cfg.optimizer)
    optimizer= c(model.parameters(), **vars(cfg.opt_cfg))
    return optimizer

def get_scheduler(optimizer, cfg, n_steps):
    if cfg.scheduler is None:
        return None
    elif cfg.scheduler == "CosineAnnealingLR":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max = n_steps,
            eta_min = cfg.lr_min,
            )
    else:
        raise ValueError(f"{cfg.scheduler} is not a valid scheduler.")

def flatten_dict(d):
    def _flatten(current_key, nested_dict, flattened_dict):
        for k, v in nested_dict.items():
            new_key = f"{current_key}.{k}" if current_key else k
            if isinstance(v, dict) and v:
                _flatten(new_key, v, flattened_dict)
            elif v is not None and v != {}:  # Exclude None values and empty dictionaries
                flattened_dict[new_key] = v
    
    flattened_dict = {}
    _flatten("", d, flattened_dict)
    return flattened_dict

def save_weights(model, cfg, epoch):
    # Save weights
    fpath= "./data/{}/{}_seed{}_epoch{}.pt".format(
        "models" if not cfg.pretrain else "models_pretrained",
        cfg.config_file, 
        cfg.seed, 
        epoch,
        )
    torch.save(model.state_dict(), fpath)
    print("SAVED WEIGHTS: ", fpath)

    # Save cfg
    fpath= fpath.replace(".pt", ".pkl")
    with open(fpath, 'wb') as f:
        pickle.dump(cfg, f)
    
    return

def is_kaggle_env():
    return 'KAGGLE_URL_BASE' in os.environ

def log_elapsed(start_time, msg=""):
    end_time= time.time()
    elapsed = (end_time - start_time)
    print(f"{msg}: {int(elapsed // 3600)}h {int(elapsed % 3600 // 60)}m {int(elapsed % 60)}s")
    return