import torch
import torch.nn as nn
from .discriminator import TemporalDiscriminator, SpatialDiscriminator
from .generator import Sampler, ContextConditionStack, LatentConditionStack

class Generator(nn.Module):
    def __init__(self, tstep):
        super().__init__()
        ## (batch, H, W)
        self.latentStack = LatentConditionStack((8, 8, 8))

        self.contextStack = ContextConditionStack()
        
        self.sampler = Sampler(tstep=tstep, chs=768, up_step=4)

    def forward(self, x):
        """
        x: input seq -> dims (N, D, C, H, W)
        """
        context_inits = self.contextStack(x)
        zlatent = self.latentStack(batch_size=1)
        pred = self.sampler(zlatent, context_inits)

        return pred

class Discriminator(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.spatial = SpatialDiscriminator(in_channel=in_channel)
        self.temporal = TemporalDiscriminator(in_channel=in_channel)

    def forward(self, x):
        """
        input_seq -> dims (N,D, C, H, W)
        """
        spatial_out = self.spatial(x)
        temporal_out = self.temporal(x)

        dis_out = torch.cat([spatial_out, temporal_out], dim=1)

        return dis_out

if __name__ == '__main__':
    ## params 
    tstep = 4
    fstep = 8
    batch_size = 10
    in_c = 1
    ## make model
    gen = Generator(tstep=tstep)
    dis = Discriminator(in_channel=in_c)
    ## fake input
    ## (N, D, C, H, W)
    fake_input = torch.rand(batch_size, tstep, in_c, 256, 256)
    fake_label = torch.rand(batch_size, fstep, in_c, 256, 256)

    gen_output = gen(fake_input)
    print(gen_output.shape)

    dis_output = dis(fake_label)
    print(dis_output.shape)
