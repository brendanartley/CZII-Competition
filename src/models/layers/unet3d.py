from typing import Type, Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.blocks import UpSample, SubpixelUpsample

class ConvBnAct3d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding: int = 0,
        stride: int = 1,
        norm_layer: Type[nn.Module] = nn.BatchNorm3d,
        act_layer: Type[nn.Module] = nn.ReLU,
    ):
        super().__init__()

        self.conv= nn.Conv3d(
            in_channels, 
            out_channels,
            kernel_size,
            stride=stride, 
            padding=padding, 
            bias=False,
        )
        self.norm = norm_layer(out_channels)
        self.act= act_layer(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class SCSEModule3d(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(in_channels, in_channels // reduction, 1),
            nn.Tanh(),
            nn.Conv3d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(
            nn.Conv3d(in_channels, 1, 1), 
            nn.Sigmoid(),
            )

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)

class Attention3d(nn.Module):
    def __init__(self, name, **params):
        super().__init__()
        if name is None:
            self.attention = nn.Identity(**params)
        elif name == "scse":
            self.attention = SCSEModule3d(**params)
        else:
            raise ValueError("Attention {} is not implemented".format(name))

    def forward(self, x):
        return self.attention(x)

class AttentionGate3d(nn.Module):
    def __init__(
        self, 
        in_channels, 
        gate_channels, 
        inter_channels: int = None,
        act_layer: Type[nn.Module] = nn.Tanh,
    ):
        super().__init__()

        if inter_channels is None:
            inter_channels = min(in_channels, gate_channels)

        self.conv_in = nn.Conv3d(in_channels, inter_channels, kernel_size=1, stride=1, padding=0)
        self.conv_gate = nn.Conv3d(gate_channels, inter_channels, kernel_size=1, stride=1, padding=0)
        
        try:
            self.act = act_layer(inplace=True)
        except:
            self.act = act_layer()

        self.attn = nn.Conv3d(inter_channels, 1, kernel_size=1, stride=1, padding=0)
        self.sig = nn.Sigmoid()

    def forward(self, x, skip):
        x_out = self.conv_in(x)
        skip_out = self.conv_gate(skip)
        
        a = self.act(skip_out + x_out)
        a = self.sig(self.attn(a))
        skip = skip * a
        return skip
        

class DecoderBlock3d(nn.Module):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        norm_layer: Type[nn.Module] = nn.BatchNorm3d,
        act_layer: Type[nn.Module] = nn.ReLU,
        attention_type: str = None,
        attention_gate: bool = False,
        upsample_mode: str = "nontrainable",
        separable: bool = False,
    ):
        super().__init__()
        
        # Upsample block
        if upsample_mode == "pixelshuffle":
            if separable:
                conv_block= nn.Sequential(
                    nn.Conv3d(in_channels, in_channels, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), groups=in_channels),
                    nn.Conv3d(in_channels, in_channels*8, kernel_size=(1, 1, 1)),
                )
            else:
                conv_block = "default"
            self.upsample= SubpixelUpsample(
                spatial_dims= 3,
                in_channels= in_channels,
                scale_factor= 2,
                conv_block= conv_block,
            )
        else:
            self.upsample = UpSample(
                spatial_dims= 3,
                in_channels= in_channels,
                out_channels= in_channels,
                scale_factor= (2, 2, 2),
                mode= upsample_mode,
            )

        # Attentions and Convs
        if attention_gate and skip_channels != 0:
            self.attention_gate = AttentionGate3d(
                in_channels, 
                skip_channels,
                )
        else:
            self.attention_gate= None

        self.attention1 = Attention3d(
            name= attention_type, 
            in_channels= in_channels + skip_channels,
            )

        self.conv1 = ConvBnAct3d(
            in_channels + skip_channels,
            out_channels,
            kernel_size= 3,
            padding= 1,
            norm_layer= norm_layer,
        )

        self.conv2 = ConvBnAct3d(
            out_channels,
            out_channels,
            kernel_size= 3,
            padding= 1,
            norm_layer= norm_layer,
        )
        self.attention2 = Attention3d(
            name= attention_type, 
            in_channels= out_channels,
            )

    def forward(self, x, skip: Optional[torch.Tensor] = None):
        x = self.upsample(x)

        if skip is not None:
            
            if self.attention_gate is not None:
                skip = self.attention_gate(x, skip)

            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x


class UnetDecoder3d(nn.Module):
    def __init__(
        self,
        encoder_channels: Tuple,
        skip_channels: Tuple = None,
        decoder_channels: Tuple = (256, 128, 64, 32, 16),
        norm_layer: Type[nn.Module] = nn.BatchNorm3d,
        act_layer: Type[nn.Module] = nn.ReLU,
        attention_type: str = None,
        attention_gate: bool = False,
        upsample_mode: str = "nontrainable",
        separable: bool = False,
    ):
        super().__init__()
        
        if len(encoder_channels) == 4:
            decoder_channels= decoder_channels[1:]
        self.decoder_channels= decoder_channels
        
        if skip_channels is None:
            skip_channels= list(encoder_channels[1:]) + [0]

        # Build decoder blocks
        in_channels= [encoder_channels[0]] + list(decoder_channels[:-1])
        self.blocks = nn.ModuleList()

        for i, (ic, sc, dc) in enumerate(zip(in_channels, skip_channels, decoder_channels)):

            # Avoid full pixel shuffle in the deepest block to limit params
            if i == 0 and upsample_mode == "pixelshuffle" and not separable:
                sep = True
            else:
                sep = separable

            self.blocks.append(
                DecoderBlock3d(
                    ic, sc, dc, 
                    norm_layer= norm_layer,
                    act_layer= act_layer,
                    attention_type= attention_type,
                    attention_gate= attention_gate,
                    upsample_mode= upsample_mode,
                    separable= sep,
                    )
            )

    def forward(self, feats: List[torch.Tensor]):
        res= [feats[0]]
        feats= feats[1:]

        # Decoder blocks
        for i, b in enumerate(self.blocks):
            skip= feats[i] if i < len(feats) else None
            res.append(
                b(res[-1], skip=skip),
                )
            
        return res

class SegmentationHead3d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        size: Tuple[int, int, int],
        kernel_size: int = 3,
        mode: str = "trilinear",
    ):
        super().__init__()
        self.size= size
        self.conv= nn.Conv3d(
            in_channels, out_channels, kernel_size= kernel_size,
            padding= kernel_size//2
        )
        self.upsample = nn.Upsample(size=self.size, mode=mode, align_corners=False)

    def forward(self, x):
        x= self.conv(x)
        if x.shape[-3:] != self.size:
            x = self.upsample(x)
        return x


if __name__ == "__main__":

    m= UnetDecoder3d(
        encoder_channels=[128, 64, 32, 16, 8],
    )
    m.cuda().eval()

    with torch.no_grad():
        x= [
            torch.ones([2, 128, 1, 4, 4]).cuda(), 
            torch.ones([2, 64, 2, 8, 8]).cuda(), 
            torch.ones([2, 32, 4, 16, 16]).cuda(), 
            torch.ones([2, 16, 8, 32, 32]).cuda(), 
            torch.ones([2, 8, 16, 64, 64]).cuda(),
            ]

        z = m(x)
        print([_.shape for _ in z])