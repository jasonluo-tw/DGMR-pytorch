import torch
import torch.nn as nn
from DGMR.model import Generator, Discriminator
from DGMR.common import DBlock
from DGMR.ConvGRU import ConvGRU
from DGMR.discriminator import SpatialDiscriminator
from loss import hinge_loss_dis, hinge_loss_gen, grid_cell_regularizer

#torch.autograd.set_detect_anomaly(True)

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
