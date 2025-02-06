import os
os.environ['NO_ALBUMENTATIONS_UPDATE']= "1"

from types import SimpleNamespace
import torch
import torch.nn as nn
import socket
import albumentations as A
import cv2

# General
cfg = SimpleNamespace(**{})
cfg.project= "czii"
cfg.hostname = socket.gethostname()
cfg.device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg.weights_path= ""
cfg.fast_dev_run= False
cfg.save_weights= True
cfg.logger= None # None, "wandb"
cfg.no_tqdm= False
cfg.train= True
cfg.val= True
cfg.seed= -1
cfg.fold= 0

# Scheduler
cfg.scheduler = None
cfg.lr_min= 1e-6
cfg.epochs= 36
cfg.epoch_steps= 100
cfg.epochs_checkpoint= 5
cfg.epochs_checkpoint_start= 999

# Optimizer
cfg.optimizer= "AdamW"
opt_cfg= SimpleNamespace()
opt_cfg.betas= (0.9, 0.999)
opt_cfg.lr = 1e-4
opt_cfg.weight_decay = 1e-6
cfg.opt_cfg= opt_cfg

# Dataset/Dataloader
cfg.dataset_type= "czii"
cfg.pretrain= False
cfg.num_workers= 0
cfg.drop_last= True
cfg.pin_memory = False
cfg.batch_size= 32
cfg.batch_size_val= 1

# Model
cfg.model_type = "unet3d"
cfg.backbone = "r3d50"
cfg.in_chans= 1
cfg.seg_classes= 7
cfg.ema= True
cfg.ema_decay= 0.99
cfg.deep_supervision= True
cfg.cls_head= False

# Encoder cfg
encoder_cfg= SimpleNamespace()
encoder_cfg.drop_path_rate= 0.3
cfg.encoder_cfg= encoder_cfg

# Decoder cfg
decoder_cfg= SimpleNamespace()
decoder_cfg.decoder_channels= (256, 128, 64, 32, 16)
decoder_cfg.upsample_mode= "pixelshuffle"
decoder_cfg.separable= False
cfg.decoder_cfg= decoder_cfg

# Loss cfg
cfg.loss_type= "DiceCELoss"
loss_cfg= SimpleNamespace()
loss_cfg.include_background= True
loss_cfg.to_onehot_y= False
loss_cfg.softmax= True
loss_cfg.sigmoid= False
loss_cfg.label_smoothing= 1e-4
cfg.loss_cfg= loss_cfg

# Monai infer cfg
infer_cfg= SimpleNamespace()
infer_cfg.sw_batch_size= 32
infer_cfg.mode= "constant"
infer_cfg.progress= False
infer_cfg.overlap= (0.5, 0.5, 0.5)
cfg.infer_cfg= infer_cfg

# Augs
cfg.patch_size= (32,128,128)
cfg.cutmix_prob= 1.0
cfg.cutmix_skipz= True
cfg.mixup_prob= 0.0
cfg.mixup_alpha= 10.0
cfg.cutpaste_prob= 1.0
cfg.rotate_prob= 0.15

# Other
cfg.local_rank= 0
cfg.mixed_precision= True
cfg.grad_accumulation= 1
cfg.track_grad_norm= True
cfg.grad_clip= 5.0
cfg.grad_norm_type= 2
cfg.logging_steps= 100