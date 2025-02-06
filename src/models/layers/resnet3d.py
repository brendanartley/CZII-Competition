from typing import Tuple
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.layers import DropPath
from torchvision.ops import DropBlock3d

from .utils import load_weights

def conv3x3x3(ic, oc, stride=1):
    return nn.Conv3d(
        ic,
        oc,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
        )

class BasicBlock(nn.Module):
    def __init__(
        self, 
        ic, 
        oc, 
        stride: int = 1, 
        downsample: bool = None, 
        expansion_factor: int = 1,
        drop_path_rate: float = 0.2,
        drop_block_rate: float = 0.2,
        drop_block_size: int = 1,
    ):
        super().__init__()
        self.conv1 = conv3x3x3(ic, oc, stride)
        self.bn1 = nn.BatchNorm3d(oc)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(oc, oc)
        self.bn2 = nn.BatchNorm3d(oc)

        self.drop_path= DropPath(drop_prob=drop_path_rate)
        self.drop_block= DropBlock3d(
            p= drop_block_rate,
            block_size= drop_block_size,
            )

        if downsample:
            self.downsample = nn.Sequential(
                nn.Conv3d(
                    ic * expansion_factor, 
                    oc,
                    kernel_size=(1, 1, 1), 
                    stride=(2,2,2), 
                    bias=False
                    ),
                nn.BatchNorm3d(oc),
            )
        else:
            self.downsample= nn.Identity()

    def forward(self, x):        
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.drop_block(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x = self.drop_path(x)

        residual = self.downsample(residual)
        x += residual
        x = self.relu(x)

        return x

class Bottleneck(nn.Module):
    def __init__(
        self, 
        ic, 
        oc, 
        stride: int = 1, 
        downsample: bool = None, 
        expansion_factor: int = 4,
        drop_path_rate: float = 0.2,
        drop_block_rate: float = 0.2,
        drop_block_size: int = 1,
    ):
        super().__init__()

        self.conv1 = nn.Conv3d(ic * expansion_factor, oc, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(oc)
        self.conv2 = nn.Conv3d(oc, oc, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(oc)
        self.conv3 = nn.Conv3d(oc, oc * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(oc * 4)
        self.relu = nn.ReLU(inplace=True)

        self.drop_path= DropPath(drop_prob=drop_path_rate)
        self.drop_block= DropBlock3d(
            p= drop_block_rate,
            block_size= drop_block_size,
            )

        if downsample is not None:
            stride = (1,1,1) if expansion_factor == 1 else (2,2,2)
            self.downsample = nn.Sequential(
                nn.Conv3d(ic * expansion_factor, oc * 4, kernel_size=(1, 1, 1), stride=stride, bias=False),
                nn.BatchNorm3d(oc * 4),
            )
        else:
            self.downsample= nn.Identity()

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.drop_block(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        x = self.drop_path(x)

        residual = self.downsample(residual)
        x += residual
        x = self.relu(x)

        return x


class ResnetEncoder3d(nn.Module):
    def __init__(
        self, 
        cfg: SimpleNamespace,
        inference_mode: bool = False,
        drop_path_rate: float = 0.2,
        drop_block_rate: float = 0.2,
        drop_block_size: int = 1,
        in_stride: Tuple[int]= (2,2,2),
        in_dilation: Tuple[int]= (1,1,1),
    ):
        super().__init__()
        self.cfg= cfg

        # Backbone init cfg
        bb= self.cfg.backbone
        backbone_cfg= {
            "r3d18": ([2, 2, 2, 2], BasicBlock),
            "r3d34": ([3, 4, 6, 3], BasicBlock),
            "r3d50": ([3, 4, 6, 3], Bottleneck),
        }
        if bb in backbone_cfg:
            layers, block = backbone_cfg[bb]
            wpath = "./data/model_zoo/{}_KM_200ep.pt".format(bb)
        else:
            raise ValueError(f"ResnetEncoder3d backbone: {bb} not implemented.")

        # Drop_path_rates (linearly scaled)
        num_blocks = sum(layers)
        flat_drop_path_rates = [drop_path_rate * (i / (num_blocks - 1)) for i in range(num_blocks)]
        drop_path_rates = []
        start = 0
        for b in layers:
            end = start + b
            drop_path_rates.append(flat_drop_path_rates[start:end])
            start = end

        # Drop_block_rates (linearly scaled)
        num_blocks = sum(layers)
        flat_drop_block_rates = [drop_block_rate * (i / (num_blocks - 1)) for i in range(num_blocks)]
        drop_block_rates = []
        start = 0
        for b in layers:
            end = start + b
            drop_block_rates.append(flat_drop_block_rates[start:end])
            start = end

        # Stem
        in_padding= tuple(_*3 for _ in in_dilation)
        self.conv1 = nn.Conv3d(
            in_channels= 3, 
            out_channels= 64,
            kernel_size= (7, 7, 7), 
            stride= in_stride, 
            dilation= in_dilation,
            padding= in_padding, 
            bias= False,
            )
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)

        # Blocks
        self.layer1 = self._make_layer(
            ic=64, oc=64, block=block, n_blocks=layers[0], stride=1, downsample=False, 
            drop_path_rates= drop_path_rates[0], drop_block_rates= drop_block_rates[0],
            )

        self.layer2 = self._make_layer(
            ic=64, oc=128, block=block, n_blocks=layers[1], stride=2, downsample=True,
            drop_path_rates=drop_path_rates[1], drop_block_rates= drop_block_rates[1],
            )

        self.layer3 = self._make_layer(
            ic=128, oc=256, block=block, n_blocks=layers[2], stride=2, downsample=True,
            drop_path_rates=drop_path_rates[2], drop_block_rates= drop_block_rates[2],
            )

        self.layer4 = self._make_layer(
            ic=256, oc=512, block=block, n_blocks=layers[3], stride=2, downsample=True,
            drop_path_rates=drop_path_rates[3], drop_block_rates= drop_block_rates[3],
            )

        # NOTE: Get weights from here: https://github.com/kenshohara/3D-ResNets-PyTorch
        # # Load pretrained weights
        # if not inference_mode:
        #     load_weights(self, wpath)

        # In channels
        self._update_input_channels()

        # Encoder channels
        with torch.no_grad():
            out = self.forward_features(torch.randn((1, self.cfg.in_chans, 32, 64, 64)))
            self.channels = [o.shape[1] for o in out]
            del out

    def _make_layer(
        self, ic, oc, block, n_blocks, stride=1, downsample=False, 
        drop_path_rates=[], drop_block_rates=[],
        ):
        layers = []
        if downsample:
            layers.append(
                block(
                    ic=ic, oc=oc, stride=stride, downsample=downsample, 
                    drop_path_rate=drop_path_rates[0], drop_block_rate=drop_block_rates[0],
                    ),
                )
        else:
            layers.append(
                block(
                    ic=ic, oc=oc, stride=stride, downsample=downsample, expansion_factor=1,
                    drop_path_rate=drop_path_rates[0], drop_block_rate=drop_block_rates[0],
                    ),
                )
        
        for i in range(1, n_blocks):
            layers.append(block(oc, oc, drop_path_rate=drop_path_rates[i], drop_block_rate=drop_block_rates[i]))

        return nn.Sequential(*layers)

    def _update_input_channels(self, ):
        with torch.no_grad():
            # Get stem
            b= self.conv1

            # Update channels
            ic= self.cfg.in_chans
            b.in_channels = ic
            w = b.weight.sum(dim=1, keepdim=True) / ic
            b.weight = nn.Parameter(w.repeat([1, ic] + [1] * (w.ndim - 2)))
        return

    def forward_features(self, x):
        res= []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        res.append(x)

        x = self.maxpool(x)
        
        x = self.layer1(x)
        res.append(x)
        x = self.layer2(x)
        res.append(x)
        x = self.layer3(x)
        res.append(x)
        x = self.layer4(x)
        res.append(x)

        return res

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


if __name__ == "__main__":
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    cfg= SimpleNamespace()
    cfg.backbone= "r3d50"
    cfg.in_chans= 1
    cfg.encoder_cfg= SimpleNamespace()

    m = ResnetEncoder3d(
        cfg= cfg,
        inference_mode= False,
        **vars(cfg.encoder_cfg),
    ).eval()

    # Param count
    n_params= count_parameters(m)
    print(f"Model: {type(m).__name__}")
    print("n_param: {:_}".format(n_params))

    # Fpass
    x= torch.ones(8, cfg.in_chans, 32, 128, 128)
    print(x.shape)

    z= m.forward_features(x)
    print([_.shape for _ in z])
