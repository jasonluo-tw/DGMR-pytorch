import torch
import torch.nn as nn
from DGMR.model import Generator, Discriminator
from DGMR.common import DBlock
from DGMR.ConvGRU import ConvGRU
from DGMR.generator import ContextConditionStack, LatentConditionStack, Sampler
from DGMR.discriminator import SpatialDiscriminator
from loss import hinge_loss_dis, hinge_loss_gen, grid_cell_regularizer

#torch.autograd.set_detect_anomaly(True)
base_c = 24
down_step = 4
time_step = 4
in_channels = 1
width = 256
height = 256
###
out_channels = base_c * 2 **(down_step-2) * time_step * in_channels
assert width % (2*2**down_step) == 0
assert height % (2*2**down_step) == 0

def test_contextstack():
    con = ContextConditionStack(
            in_channels=in_channels,
            base_channels=base_c,
            down_step=down_step,
            prev_step=time_step
    )
    x = torch.rand(10, 4, 1, 256, 256)
    y = con(x)
    for i in y:
        print(i.shape)

def test_latentstack():
    s_w = width // (2 * 2**down_step)
    s_h = height // (2 * 2**down_step)
    lat = LatentConditionStack(
            in_shape=(s_w, s_h),
            out_channels=out_channels*2,
            use_cuda=False
    )

def test_sampler():
    sam = Sampler(
            in_shape=(width, height),
            in_channels=in_channels,
            pred_step=18,
            base_channels=base_c,
            up_step=down_step
    )
    b = 2
    latent = torch.rand(1, 768, 8, 8)
    #torch.Size([10, 48, 64, 64])
    #torch.Size([10, 96, 32, 32])
    #torch.Size([10, 192, 16, 16])
    #torch.Size([10, 384, 8, 8])
    init_states = [torch.rand(b, int(48*2**i), int(64/2**i), int(64/2**i)) for i in range(4)]
    for i in init_states:
        print(i.shape)
    y = sam(latent, init_states)
    print(y.shape, '~~~~')

test_sampler()

"""
tstep = 4
fstep = 4
batch_size = 4
in_c = 1

## Get model
#gen = Generator(tstep=fstep)

fake_input = torch.ones((batch_size, 10, in_c, 8, 8))

#g_opt = torch.optim.Adam(gen.parameters())

model = DBlock(in_channel=1, out_channel=2, conv_type='2d', apply_relu=True, apply_down=True)
#model = SpatialDiscriminator(in_channel=1)
#model = ConvGRU((8, 8), 1, 2, 3)

d_opt = torch.optim.Adam(model.parameters())

out = model(fake_input[:, 0, :, :, :])
out1 = model(fake_input[:, 1, :, :, :])
out = torch.stack([out, out1], dim=1)

print(out.shape, 'shape')
fake_label = torch.rand(*out.shape)
loss = torch.mean(torch.abs(out - fake_label))

d_opt.zero_grad()
loss.backward()
d_opt.step()

print('Success')
"""

"""
##
## (N, D, C, W, H)
#gen_output = gen(fake_input)

gen_seq = torch.cat([fake_input, fake_label], dim=1)
real_seq = torch.cat([fake_input, fake_label], dim=1)

# cat along batch dimension
concat_seq = torch.cat([gen_seq, real_seq], dim=0)

concat_out = dis(concat_seq)

#fake, real = torch.split(concat_out, 1, dim=1)

dis_loss = hinge_loss_dis(concat_out, concat_seq)
d_opt.zero_grad()
dis_loss.backward()

#dis_loss.backward()
#d_opt.step()

#g_opt.zero_grad()
"""
