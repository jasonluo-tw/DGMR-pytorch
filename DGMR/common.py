import numpy as np
#TODO: doesn't know why this will cause inplace problem in forloop operation which is used in DBlock
from torch.nn.utils.parametrizations import spectral_norm
#from torch.nn.utils import spectral_norm
import torch
from torch.nn import functional as F
import einops

class GBlock(torch.nn.Module):
    def __init__(self, in_channel: int, out_channel: int):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel

        self.bn1 = torch.nn.BatchNorm2d(in_channel)
        self.bn2 = torch.nn.BatchNorm2d(in_channel)

        self.relu = torch.nn.ReLU()
        ## conv1x1
        self.conv1x1 = spectral_norm(
            torch.nn.Conv2d(in_channel, out_channel, kernel_size=1)
        )

        self.conv3x3_1 = spectral_norm(
            torch.nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1)
        )
        self.conv3x3_2 = spectral_norm(
            torch.nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1)
        )

    def forward(self, x: torch.Tensor):
        ## if shape is different then applied
        if x.shape[1] != self.out_channel:
            res = self.conv_1x1(x)
        else:
            res = x.clone()

        ## first 
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv3x3_1(x)
        ## second
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3x3_2(x)

        y = x + res

        return y

class Up_GBlock(torch.nn.Module):
    def __init__(self, in_channel: int):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = int(in_channel / 2)

        self.bn1 = torch.nn.BatchNorm2d(in_channel)
        #self.bn2 = torch.nn.BatchNorm2d(self.out_channel)
        self.bn2 = torch.nn.BatchNorm2d(in_channel)

        self.relu = torch.nn.ReLU()
        self.up = torch.nn.Upsample(scale_factor=2, mode='nearest')

        self.conv1x1 = spectral_norm(
            torch.nn.Conv2d(in_channel, self.out_channel, kernel_size=1)
        )

        self.conv3x3_1 = spectral_norm(
            torch.nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1)
        )
        self.conv3x3_2 = spectral_norm(
            torch.nn.Conv2d(in_channel, self.out_channel, kernel_size=3, padding=1)
        )

    def forward(self, x):
        res = self.up(x)
        res = self.conv1x1(res)

        x = self.bn1(x)
        x = self.relu(x)
        x = self.up(x)
        x = self.conv3x3_1(x)

        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3x3_2(x)

        y = x + res

        return y

class DBlock(torch.nn.Module):
    def __init__(self, in_channel: int, out_channel: int, conv_type='2d', apply_relu=True, apply_down=False):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.apply_relu = apply_relu
        self.apply_down = apply_down
        
        ## construct layer
        if conv_type == '2d':
            self.avg_pool = torch.nn.AvgPool2d(kernel_size=2, stride=2)
            conv = torch.nn.Conv2d
        elif conv_type == '3d':
            self.avg_pool = torch.nn.AvgPool3d(kernel_size=2, stride=2)
            conv = torch.nn.Conv3d

        self.relu = torch.nn.ReLU()
        self.conv1x1 = spectral_norm(
            conv(in_channel, out_channel, kernel_size=1)
        )

        self.conv3x3_1 = spectral_norm(
            conv(in_channel, out_channel, kernel_size=3, padding=1)
        )
        self.conv3x3_2 = spectral_norm(
            conv(out_channel, out_channel, kernel_size=3, padding=1)
        )

    def forward(self, x):
        ## Residual block
        if x.shape[1] != self.out_channel:
            res = self.conv1x1(x)
        else:
            res = x.clone()
        if self.apply_down:
            res = self.avg_pool(res)

        ##
        if self.apply_relu:
            x = self.relu(x)
        x = self.conv3x3_1(x)
        x = self.relu(x)
        x = self.conv3x3_2(x)
        if self.apply_down:
            x = self.avg_pool(x)

        ## connect
        y = res + x

        return y

class LBlock(torch.nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel

        self.relu = torch.nn.ReLU()
        conv = torch.nn.Conv2d
        self.conv1x1 = conv(in_channel, (out_channel-in_channel), kernel_size=1)

        self.conv3x3_1 = conv(in_channel, in_channel, kernel_size=3, padding=1)
        self.conv3x3_2 = conv(in_channel, out_channel, kernel_size=3, padding=1)

    def forward(self, x):
        res = torch.cat([x, self.conv1x1(x)], dim=1)

        x = self.relu(x)
        x = self.conv3x3_1(x)
        x = self.relu(x)
        x = self.conv3x3_2(x)

        y = x + res

        return y

## Attention Layer
def attention_einsum(q, k, v):
    """
    Apply self-attention to tensors
    """
    
    ## Reshape 3D tensor to 2D tensor with first dimension L = h x w
    k = einops.rearrange(k, "h w c -> (h w) c") # [h, w, c] -> [L, c]
    v = einops.rearrange(v, "h w c -> (h w) c") # [h, w, c] -> [L, c]

    # Einstein summation corresponding to the query * key operation.
    beta = F.softmax(torch.einsum("hwc, Lc->hwL", q, k), dim=-1)

    # Einstein summation corresponding to the attention * value operation.
    out = torch.einsum("hwL, Lc->hwc", beta, v)

    return out

class AttentionLayer(torch.nn.Module):
    def __init__(self, in_channel, out_channel, ratio_kq=8, ratio_v=8):
        super().__init__()

        self.ratio_kq = ratio_kq
        self.ratio_v = ratio_v
        self.in_channel = in_channel
        self.out_channel = out_channel

        ## compute query, key, and value using 1x1 convolution
        self.query = torch.nn.Conv2d(
            in_channel,
            out_channel//ratio_kq,
            kernel_size=1,
            bias=False
        )
        
        self.key = torch.nn.Conv2d(
            in_channel,
            out_channel//ratio_kq,
            kernel_size=1,
            bias=False
        )

        self.value = torch.nn.Conv2d(
            in_channel,
            out_channel//ratio_v,
            kernel_size=1,
            bias=False
        )

        self.conv = torch.nn.Conv2d(
            out_channel//8,
            out_channel,
            kernel_size=1,
            bias=False
        )
        
        self.gamma = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        query = self.query(x)
        key   = self.key(x)
        value = self.value(x)
        ## apply attention
        out = []
        for i in range(x.shape[0]):
            out.append(attention_einsum(query[i], key[i], value[i]))

        out = torch.stack(out, dim=0)
        out = self.gamma * self.conv(out)
        out = out + x  ## skip connection

        return out
