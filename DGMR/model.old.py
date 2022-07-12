import torch
import torch.nn as nn
from .discriminator import TemporalDiscriminator, SpatialDiscriminator
from .generator import Sampler, ContextConditionStack, LatentConditionStack

class Generator(nn.Module):
    def __init__(
            self,
            in_channels,
            in_shape,
            base_channels,
            down_step,
            prev_step,
            pred_step,
            batch_size,
            use_cuda=True,
    ):
        super().__init__()
        width  = in_shape[0]
        height = in_shape[1]
        assert width % (2*2**down_step) == 0
        assert height % (2*2**down_step) == 0
        s_w = width // (2 * 2**down_step)
        s_h = height // (2 * 2**down_step)
        out_channels = base_channels * 2**(down_step-2) * prev_step * in_channels
        ## (batch, H, W)

        self.latentStack = LatentConditionStack(
                in_shape = (s_w, s_h),
                out_channels = out_channels*2,
                use_cuda = use_cuda
        )
       
        self.contextStack = ContextConditionStack(
                in_channels = in_channels,
                base_channels = base_channels,
                down_step = down_step,
                prev_step = prev_step
        )
        
        self.sampler = Sampler(
                in_shape = (width, height),
                in_channels = in_channels,
                pred_step = pred_step,
                base_channels = base_channels,
                up_step = down_step
        )

    def forward(self, x):
        """
        x: input seq -> dims (N, D, C, H, W)
        """
        context_inits = self.contextStack(x)
        zlatent = self.latentStack(x, batch_size=1)
        pred = self.sampler(zlatent, context_inits)

        return pred

class Discriminator(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.spatial = SpatialDiscriminator(in_channel=in_channels)
        self.temporal = TemporalDiscriminator(in_channel=in_channels)

    def forward(self, input_seq, idx):
        """
        input_seq -> dims (N, D, C, H, W) e.g. input_frames + pred(target)_frames
        idx -> the starting index of prediction frame
        """
        ff = [i for i in range(idx, input_seq.shape[1])]
        spatial_out = self.spatial(input_seq[:, ff, :, :, :])

        temporal_out = self.temporal(input_seq)

        dis_out = torch.cat([spatial_out, temporal_out], dim=1)

        return dis_out

