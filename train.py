import argparse
from copy import copy
import sys
from importlib import import_module
import os
import random
import numpy as np
import torch

from src.utils import update_cfg

def set_seed(seed=1234):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-C", "--config", help="config filename", default="r3d34")
    parser.add_argument("-G", "--gpu_id", default="", help="GPU ID")
    parser_args, other_args = parser.parse_known_args(sys.argv)

    # Use all GPUs unless specified
    if parser_args.gpu_id != "":
        os.environ['CUDA_VISIBLE_DEVICES'] = str(parser_args.gpu_id)

    # Load CFG
    cfg = copy(import_module('src.configs.{}'.format(parser_args.config)).cfg)
    cfg.config_file = parser_args.config
    print("config ->", cfg.config_file)

    # Update args
    if len(other_args) > 1:
        other_args = {v.split("=")[0].lstrip("-"):v.split("=")[1] for v in other_args[1:]}
        cfg= update_cfg(
            cfg=cfg, other_args=other_args, log=True,
            )

    # Set seed
    if cfg.seed < 0:
        cfg.seed = np.random.randint(1_000_000)
    print("seed", cfg.seed)
    set_seed(cfg.seed)

    if cfg.fast_dev_run:
        cfg.epochs= 1
        cfg.no_wandb= None

    if cfg.pretrain:
        cfg.fold= -1
        cfg.epochs_checkpoint= 999
        cfg.epochs= 20
        cfg.epoch_steps= 250
        cfg.rotate_prob= 0.0
    
    return cfg

if __name__ == "__main__":
    from src.modules.train import train
    cfg = parse_args()
    train(cfg)