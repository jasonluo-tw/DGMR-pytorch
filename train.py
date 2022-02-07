import torch
import torch.nn as nn
from DGMR.model import Generator, Discriminator
from DGMR.common import DBlock
from loss import hinge_loss_dis, hinge_loss_gen, grid_cell_regularizer

#torch.autograd.set_detect_anomaly(True)

tstep = 4
fstep = 4
batch_size = 2
in_c = 1

## Get model
gen = Generator(tstep=fstep)
dis = Discriminator(in_channel=in_c)

fake_input = torch.ones((batch_size, tstep, in_c, 256, 256))
fake_label = torch.ones((batch_size, fstep, in_c, 256, 256))

g_opt = torch.optim.Adam(gen.parameters())
d_opt = torch.optim.Adam(dis.parameters())

##
### (N, D, C, W, H)
#gen_output = gen(fake_input)
#
#gen_seq = torch.cat([fake_input, gen_output], dim=1)
#real_seq = torch.cat([fake_input, fake_label], dim=1)
#
## cat along batch dimension
#concat_seq = torch.cat([gen_seq, real_seq], dim=0)
#concat_out = dis(concat_seq)
#
#fake, real = torch.split(concat_out, 1, dim=1)
#
#dis_loss = hinge_loss_dis(fake, real)
#d_opt.zero_grad()
#dis_loss.backward()
#d_opt.step()

print('update generator')
## for generator's update
print(gen(fake_input).shape)

pred = [gen(fake_input) for _ in range(2)]
reg = grid_cell_regularizer(torch.stack(pred, dim=0), fake_label)

##
fake_seq = [torch.cat([fake_input, x], dim=1) for x in pred]
real_seq = torch.cat([fake_input, fake_label], dim=1)

print('update generator2')
dis_scores = []
for g_seq in fake_seq:
    concat_inputs = torch.cat([real_seq, g_seq], dim=0)
    dis_outputs = dis(concat_inputs)
    real, fake = torch.split(dis_outputs, 1, dim=1)
    dis_scores.append(fake)

g_hinge_loss = hinge_loss_gen(torch.cat(dis_scores, dim=0))
g_loss = g_hinge_loss + 20 * reg

print('update generator3')
g_opt.zero_grad()
g_loss.backward()
g_opt.step()
