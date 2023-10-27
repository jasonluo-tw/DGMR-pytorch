import torch
import torch.nn as nn
from .architect.discriminator import TemporalDiscriminator, SpatialDiscriminator
from .architect.generator import Sampler, ContextConditionStack, LatentConditionStack

class Generator(nn.Module):
    def __init__(
            self,
            in_channels,
            base_channels,
            down_step,
            prev_step,
            batch_size,
            use_cuda=True,
    ):
        super().__init__()
        out_channels = base_channels * 2**(down_step-2) * prev_step * in_channels
        self.batch_size = batch_size

        self.latentStack = LatentConditionStack(
                out_channels = out_channels*2,
                down_step = down_step,
                use_cuda = use_cuda
        )
       
        self.contextStack = ContextConditionStack(
                in_channels = in_channels,
                base_channels = base_channels,
                down_step = down_step,
                prev_step = prev_step
        )
        
        self.sampler = Sampler(
                in_channels = in_channels,
                base_channels = base_channels,
                up_step = down_step
        )

    def forward(self, x, pred_step=12):
        """
        x: input seq -> dims (N, D, C, H, W)
        """
        context_inits = self.contextStack(x)
        batch_size = context_inits[0].shape[0]
        zlatent = self.latentStack(x, batch_size=batch_size)
        pred = self.sampler(zlatent, context_inits, pred_step)

        return pred

class Discriminator(nn.Module):
    def __init__(self, in_channels, base_channels):
        super().__init__()
        self.spatial = SpatialDiscriminator(in_channel=in_channels, base_c=base_channels)
        self.temporal = TemporalDiscriminator(in_channel=in_channels, base_c=base_channels)

    def forward(self, x, y):
        """
        x -> dims (N, D, C, H, W) e.g. input_frames
        y -> dims (N, D, C, H, W) e.g. output_grames
        """
        spatial_out  = self.spatial(y)
        temporal_out = self.temporal(torch.cat([x, y], dim=1))

        dis_out = torch.cat([spatial_out, temporal_out], dim=1)

        return dis_out

