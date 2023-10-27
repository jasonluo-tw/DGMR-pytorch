from typing import Tuple

import torch
import torch.nn as nn
#TODO: doesn't know why this(torch.nn.utils.parametrizations.spectral_norm) will cause inplace problem in forloop operation (pytorch 1.09) which is used in DBlock
#TODO: but it is OK in pytorch v1.10
from torch.nn.utils.parametrizations import spectral_norm
#from torch.nn.utils import spectral_norm
from torch.distributions import normal
import einops

from .ConvGRU import ConvGRU
from .common import GBlock, Up_GBlock, LBlock, AttentionLayer, DBlock

class Sampler(nn.Module):
    def __init__(self, in_channels, base_channels=24, up_step=4):
        """
        up_step should be the same as down_step in context-condition-stack

        """
        super().__init__()
        base_c = base_channels

        self.up_steps = up_step
        self.convgru_list = nn.ModuleList()
        self.conv1x1_list = nn.ModuleList()
        self.gblock_list  = nn.ModuleList()
        self.upg_list     = nn.ModuleList()

        for i in range(self.up_steps):
            ## different scale
            chs1 = base_c * 2**(self.up_steps-i+1) * in_channels
            chs2 = base_c * 2**(self.up_steps-i) * in_channels
            ## convgru 
            self.convgru_list.append(
                ConvGRU(chs1, chs2, 3)
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
            ##TODO: close Batch
            #self.bn = nn.BatchNorm2d(chs2)
            self.relu = nn.ReLU()
            self.last_conv1x1 = spectral_norm(
                nn.Conv2d(in_channels=chs2,
                          out_channels=4,
                          kernel_size=(1, 1))
            )
            self.depth_to_space = nn.PixelShuffle(upscale_factor=2)

    def forward(self, latents, init_states, pred_step):
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
            latents, "b d c h w -> b (repeat d) c h w", repeat=pred_step
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
            ## output: seq_out list dim -> D * [N, C, H, W]
            ## should stack at dim == 1 to become (N, D, C, H, W)
            seq_out = torch.stack(seq_out, dim=1)

        ## final output
        ## forloop time step
        output = []
        for t in range(seq_out.shape[1]):
            y = seq_out[:, t, :, :, :]
            #y = self.bn(y)
            y = self.relu(y)
            y = self.last_conv1x1(y)
            y = self.depth_to_space(y)
            output.append(y)

        output = torch.stack(output, dim=1)

        return output

##TODO: Now only generate one sample, and not one batch
class LatentConditionStack(nn.Module):
    def __init__(self, out_channels, down_step, use_cuda, attn=True):
        """
        in_shape dims -> e.g. (8, 8) -> (H, W)
        x) base_c is 1/96 of out_channels
        base_c is set to 4
        """
        super().__init__()

        self.down_step = down_step
        self.base_c = out_channels // 96
        if self.base_c < 4:
            self.base_c = 4

        self.out_channels = out_channels
        self.attn = attn
        self.use_cuda = use_cuda

        ## define the distribution
        self.dist = normal.Normal(loc=0.0, scale=1.0)

        self.conv3x3 = spectral_norm(
            nn.Conv2d(
                in_channels=self.base_c,
                out_channels=self.base_c,
                kernel_size=(3, 3),
                padding=1
            )
        )

        cc = self.base_c
        self.l1 = LBlock(cc, cc*3)
        self.l2 = LBlock(cc*3, cc*6)
        self.l3 = LBlock(cc*6, cc*24)
        if self.attn:
            self.attn = AttentionLayer(cc*24,cc*24)
        self.l4 = LBlock(cc*24, self.out_channels)

    def forward(self, x, batch_size=1, z=None):
        """
        x shape -> (batch_size, time, c, width, height)
        """
        width = x.shape[3]
        height = x.shape[4]
        ## shape after downstep
        s_w = width // (2 * 2**self.down_step)
        s_h = height // (2 * 2**self.down_step)

        in_shape = [self.base_c] + [s_w, s_h]

        target_shape = [batch_size] + in_shape
        if z is None:
            z = self.dist.sample(target_shape)

        if self.use_cuda:
            #z = z.to("cuda")
            ## with lightening
            z = z.type_as(x)
        
        ## first conv
        z = self.conv3x3(z)

        ## Lblock
        z = self.l1(z)
        z = self.l2(z)
        z = self.l3(z)
        if self.attn:
            z = self.attn(z)

        z = self.l4(z)

        return z

##TODO: modification(Change the amount of parameters)
class ContextConditionStack(nn.Module):
    def __init__(self, 
            in_channels: int = 1,
            base_channels: int = 24, 
            down_step: int = 4, 
            prev_step: int = 4):
        """
        base_channels: e.g. 24 -> output_channel: 384
        output_channel: base_c*in_c*2**(down_step-2) * prev_step
        down_step: int
        prev_step: int
        """
        super().__init__()
        self.in_channels = in_channels
        self.down_step = down_step 
        self.prev_step = prev_step
        ###
        base_c = base_channels
        in_c = in_channels
       
        ## different scales channels
        chs = [4*in_c] + [base_c*in_c*2**(i+1) for i in range(down_step)]
        
        self.space_to_depth = nn.PixelUnshuffle(downscale_factor=2)
        self.Dlist = nn.ModuleList()
        self.convList = nn.ModuleList()
        for i in range(down_step):
            self.Dlist.append(
                DBlock(in_channel=chs[i],
                       out_channel=chs[i+1],
                       apply_relu=True, apply_down=True)
            )

            self.convList.append(
                spectral_norm(
                    nn.Conv2d(in_channels=prev_step * chs[i+1],
                              out_channels=prev_step * chs[i+1] // 4,
                              kernel_size=(3, 3),
                              padding=1)
                )
            )

        ## ReLU
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        ## input dims -> (N, D, C, H, W)
        """
        x = self.space_to_depth(x)
        tsteps = x.shape[1]
        assert tsteps == self.prev_step
            
        ## different feature index represent different scale
        ## features
        ## [scale1 -> [t1, t2, t3, t4], scale2 -> [t1, t2, t3, t4], scale3 -> [....]]
        features = [[] for i in range(tsteps)]

        for st in range(tsteps):
            in_x = x[:, st, :, :, :]
            ## in_x -> (Batch(N), C, H, W)
            for scale in range(self.down_step):
                in_x = self.Dlist[scale](in_x)
                features[scale].append(in_x)
        
        out_scale = []
        for i, cc in enumerate(self.convList):
            ## after stacking, dims -> (Batch, Time, C, H, W)
            ## and mixing layer is to concat Time, C
            stacked = self._mixing_layer(torch.stack(features[i], dim=1))
            out = self.relu(cc(stacked))
            out_scale.append(out)

        return out_scale

    def _mixing_layer(self, x):
        # conver from (N, Time, C, H, W) -> (N, Time*C, H, W)
        # Then apply Conv2d
        stacked = einops.rearrange(x, "b t c h w -> b (t c) h w")
        
        return stacked

