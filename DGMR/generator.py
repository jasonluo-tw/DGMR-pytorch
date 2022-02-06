from typing import Tuple

import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm
from torch.distributions import normal
import einops

from ConvGRU import ConvGRU
from common import GBlock, Up_GBlock, LBlock, AttentionLayer, DBlock

class Sampler(nn.Module):
    def __init__(self, tstep, chs=768, up_step=4):
        super().__init__()
        self.tstep = tstep
        self.up_steps = up_step
        self.convgru_list = nn.ModuleList()
        self.conv1x1_list = nn.ModuleList()
        self.gblock_list  = nn.ModuleList()
        self.upg_list     = nn.ModuleList()
        for i in range(self.up_steps):
            ## different scale
            H_W = 8 * (i+1)
            chs1 = chs // 2**(i)
            chs2 = chs // 2**(i+1)
            ## convgru 
            self.convgru_list.append(
                ConvGRU((H_W, H_W), chs1, chs2, 3)
            )
            ## conv1x1
            self.conv1x1_list.append(
                spectral_norm(
                    nn.Conv2d(in_channels=chs2, out_channels=chs1, kernel_size=(1, 1))
                )
            )
            ## GBlock
            self.gblock_list.append(
                GBlock(in_channel=chs1, out_channel=chs1)
            )
            ## upgblock
            self.upg_list.append(
                Up_GBlock(in_channel=chs1)
            )

            ## output
            self.bn = nn.BatchNorm2d(chs2)
            self.relu = nn.ReLU()
            self.last_conv1x1 = spectral_norm(
                nn.Conv2d(in_channels=chs2,
                          out_channels=4,
                          kernel_size=(1, 1))
            )
            self.depth_to_sapce = nn.PixelShuffle(upscale_factor=2)

    def forward(self, latents, init_states):
        """
        latent dim -> (N, C, W, H)
        init_states dim -> (N, C, W, H)
        """
        ## expand time dims at axis=1
        latents = torch.unsqueeze(latents, dim=1)
        ## repeat batch_size 
        if latents.shape[0] == 1:
            ## expand batch
            latents = einops.repeat(
                latents, "b d c h w -> (repeat b) d c h w", repeat=init_states[0].shape[0]
            )
        ## repeat time step
        latents = einops.repeat(
            latents, "b d c h w -> b (repeat d) c h w", repeat=self.tstep
        )
        seq_out = latents
        ## init_states should be reversed
        ## scale up step
        for i in range(self.up_steps):
            seq_out = self.convgru_list[i](seq_out, init_states[(self.up_steps-1)-i])
            ## forloop time step
            seq_out = [self.conv1x1_list[i](h) for h in seq_out]
            seq_out = [self.gblock_list[i](h) for h in seq_out]
            seq_out = [self.upg_list[i](h) for h in seq_out]
            seq_out = torch.stack(seq_out, dim=1)

        ## final output
        ## forloop time step
        output = []
        for t in range(seq_out.shape[1]):
            y = seq_out[:, t, :, :, :]
            y = self.bn(y)
            y = self.relu(y)
            y = self.last_conv1x1(y)
            y = self.depth_to_sapce(y)
            output.append(y)

        output = torch.stack(output, dim=1)

        return output

##TODO: Now only generate one sample, and not one batch
class LatentConditionStack(nn.Module):
    def __init__(self, in_shape):
        """
        in_shape dims -> (8, 8, 8) -> (C, H, W)
        """
        super().__init__()

        self.in_shape = in_shape
        self.in_channel = in_shape[0]

        self.dist = normal.Normal(loc=0.0, scale=1.0)

        self.conv3x3 = spectral_norm(
            nn.Conv2d(
                in_channels=self.in_channel,
                out_channels=self.in_channel,
                kernel_size=(3, 3),
                padding=1
            )
        )

        self.l1 = LBlock(self.in_channel, 24)
        self.l2 = LBlock(24, 48)
        self.l3 = LBlock(48, 192)
        self.attn = AttentionLayer(192, 192)
        self.l4 = LBlock(192, 768)

    def forward(self, batch_size=1):
        target_shape = [batch_size] + [*self.in_shape]
        z = self.dist.sample(target_shape)
        
        ## first conv
        z = self.conv3x3(z)

        ## Lblock
        z = self.l1(z)
        z = self.l2(z)
        z = self.l3(z)
        z = self.attn(z)

        z = self.l4(z)

        return z

class ContextConditionStack(nn.Module):
    def __init__(self, in_channels: int = 1,
                 final_channel: int = 384):
        """
        """
        super().__init__()
        self.in_channels = in_channels
        
        self.space_to_depth = nn.PixelUnshuffle(downscale_factor=2)

        in_c = in_channels
        ## different scales channels
        chs = [4*in_c, 24*in_c, 48*in_c, 96*in_c, 192*in_c]
        self.Dlist = nn.ModuleList()
        self.convList = nn.ModuleList()
        for i in range(len(chs)-1):
            self.Dlist.append(
                DBlock(in_channel=chs[i],
                       out_channel=chs[i+1],
                       apply_relu=True, apply_down=True)
            )

            self.convList.append(
                nn.Conv2d(in_channels=4 * chs[i+1],
                          out_channels=4 * chs[i+1] // 2,
                          kernel_size=(3, 3),
                          padding=1)
            )

        ## ReLU
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ## input dims -> (N, D, C, H, W)
        x = self.space_to_depth(x)
        steps = x.shape[1]

        ## different feature index represent different scale
        features = [[] for i in range(steps)]

        for st in range(steps):
            in_x = x[:, st, :, :, :]
            ## in_x -> (N, C, H, W)
            for scale in range(4):
                in_x = self.Dlist[scale](in_x)
                features[scale].append(in_x)
        
        out_scale = []
        for i, cc in enumerate(self.convList):
            ## after stacking, dims -> (N, D, C, H, W)
            stacked = self._mixing_layer(torch.stack(features[i], dim=1))
            out = self.relu(cc(stacked))
            out_scale.append(out)

        return out_scale

    def _mixing_layer(self, x):
        # conver from (N, D, C, H, W) -> (N, D*C, H, W)
        # Then apply Conv2d
        stacked = einops.rearrange(x, "b t c h w -> b (c t) h w")
        
        return stacked

if __name__ == '__main__':
    tstep = 4
    batch_size = 10
    x_chs = 1
    ## h = 256
    ## w = 256
    ## fake input radar images -> (N, D, C, H, W)
    input_x = torch.rand(batch_size, tstep, x_chs, 256, 256)

    ## produce zlatent for Sampler
    LatentStack = LatentConditionStack((8, 8, 8))
    ## produce context latent
    ContextStack = ContextConditionStack()

    model = Sampler(tstep=tstep, chs=768, up_step=4)
