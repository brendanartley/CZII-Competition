from typing import Type, Tuple, Iterable
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from importlib import import_module

from src.models.augs import aug3d
from src.models.layers import (
    ResnetEncoder3d, UnetDecoder3d, SegmentationHead3d,
)

class Net(nn.Module):
    def __init__(
        self,
        cfg: SimpleNamespace,
        inference_mode: bool = False,
    ):
        super().__init__()
        self.cfg= cfg
        self.inference_mode= inference_mode
        self.loss_fn= self._init_loss_fn()
        
        # Backbone
        self.backbone = ResnetEncoder3d(
            cfg= cfg,
            inference_mode= inference_mode,
            **vars(cfg.encoder_cfg),
            )
        ecs= self.backbone.channels[::-1]

        self.decoder= UnetDecoder3d(
            encoder_channels= ecs,
            **vars(cfg.decoder_cfg),
        )

        self.seg_head= SegmentationHead3d(
            in_channels= self.decoder.decoder_channels[-1],
            out_channels= cfg.seg_classes,
            size= self.cfg.patch_size,
        )

        if cfg.deep_supervision:
            pz, py, px= cfg.patch_size
            self.seg_head_aux1= SegmentationHead3d(
                in_channels= self.decoder.decoder_channels[-2],
                out_channels= cfg.seg_classes,
                size= (pz//2, py//2, px//2),
            )

            self.seg_head_aux2= SegmentationHead3d(
                in_channels= self.decoder.decoder_channels[-3],
                out_channels= cfg.seg_classes,
                size= (pz//4, py//4, px//4),
            )

            self.seg_head_aux3= SegmentationHead3d(
                in_channels= self.decoder.decoder_channels[-4],
                out_channels= cfg.seg_classes,
                size= (pz//8, py//8, px//8),
            )

        if cfg.pretrain:
            timm.utils.freeze(self.backbone)

    def _init_loss_fn(self, ):
        if self.inference_mode:
            c= None
        else:
            losses= import_module("monai.losses")
            l= getattr(losses, self.cfg.loss_type)
            c= l(**vars(self.cfg.loss_cfg))
        return c

    def forward(self, batch):
        # Augs
        if self.training:
            x= batch["input"].float()
            mask= batch["target"]

            x, mask= aug3d.rotate(x, mask, p= 1.0, dims=[(-2,-1)])
            x, mask= aug3d.pixel_aug(x, mask)
            x, mask= aug3d.flip_3d(x, mask)
            x, mask= aug3d.swap_dims(x, mask)
            x, mask= aug3d.mixup_3d(x, mask, p= self.cfg.mixup_prob, alpha= self.cfg.mixup_alpha)
            x, mask= aug3d.cutmix_3d(x, mask, p= self.cfg.cutmix_prob, skipz= self.cfg.cutmix_skipz)

        else:
            x= batch
        bs,c,t,h,w = x.shape

        # Encoder
        x = self.backbone.forward_features(x)
        x = x[::-1]

        # Decoder
        x= self.decoder(x)
        x_seg= self.seg_head(x[-1])

        if self.training:
            loss= self.loss_fn(x_seg, mask)

            if self.cfg.deep_supervision:
                
                # Downsample masks w/ max pool
                mask_aux1= F.max_pool3d(mask.view(-1, t, h, w), kernel_size=2)
                mask_aux1= mask_aux1.view(bs, self.cfg.seg_classes, t//2, h//2, w//2)
                if self.cfg.loss_cfg.softmax:
                    mask_aux1= mask_aux1 / (mask_aux1.sum(dim=1, keepdim=True) + 1e-8)  
                x_seg_aux1= self.seg_head_aux1(x[-2])
                loss_aux1= self.loss_fn(x_seg_aux1, mask_aux1)

                mask_aux2= F.max_pool3d(mask.view(-1, t, h, w), kernel_size=4)
                mask_aux2= mask_aux2.view(bs, self.cfg.seg_classes, t//4, h//4, w//4) 
                if self.cfg.loss_cfg.softmax:
                    mask_aux2= mask_aux2 / (mask_aux2.sum(dim=1, keepdim=True) + 1e-8) 
                x_seg_aux2= self.seg_head_aux2(x[-3])
                loss_aux2= self.loss_fn(x_seg_aux2, mask_aux2)

                mask_aux3= F.max_pool3d(mask.view(-1, t, h, w), kernel_size=8)
                mask_aux3= mask_aux3.view(bs, self.cfg.seg_classes, t//8, h//8, w//8) 
                if self.cfg.loss_cfg.softmax:
                    mask_aux3= mask_aux3 / (mask_aux3.sum(dim=1, keepdim=True) + 1e-8) 
                x_seg_aux3= self.seg_head_aux3(x[-4])
                loss_aux3= self.loss_fn(x_seg_aux3, mask_aux3)

                loss += 0.35*loss_aux1 + 0.10*loss_aux2 + 0.01*loss_aux3

            return {
                "x": x_seg,
                "loss": loss,
            }
        else:
            return x_seg

if __name__ == "__main__":
    from src.configs.r3d18 import cfg
    from src.models.utils import count_parameters

    m= Net(cfg=cfg).cuda()
    print("n_param: {:_}".format(count_parameters(m)))
    bs= 2
    z,y,x= cfg.patch_size
    batch= {
        "input": torch.ones(bs,1,z,y,x).cuda(),
        "target": torch.rand(bs,cfg.seg_classes,z,y,x).cuda(),
        "target_cls": torch.rand(bs,cfg.seg_classes-1).cuda(),
    }
    with torch.no_grad():
        z= m(batch)
