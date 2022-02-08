import torch
import torch.nn.functional as F
from torch.autograd import Variable

class ConvGRUCell(torch.nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3):
        super().__init__()
        padding = kernel_size//2
        self.out_channel = out_channel

        self.conv1 = torch.nn.Conv2d(
            in_channels=in_channel+out_channel,
            out_channels=2*out_channel,
            kernel_size=kernel_size,
            padding=padding
        )

        self.conv2 = torch.nn.Conv2d(
            in_channels=in_channel+out_channel,
            out_channels=out_channel,
            kernel_size=kernel_size,
            padding=padding
        )

    def forward(self, x, h_st):
        xx = torch.cat([x, h_st], dim=1)
        xx = self.conv1(xx)
        gamma, beta = torch.split(xx, self.out_channel, dim=1)

        reset_gate = torch.sigmoid(gamma)
        update_gate = torch.sigmoid(beta)

        out = torch.cat([x, h_st*reset_gate], dim=1)
        out = torch.tanh(self.conv2(out))
        
        out = (1-update_gate) * out + h_st * update_gate
        new_st = out

        return out, new_st

class ConvGRU(torch.nn.Module):
    def __init__(self, image_shape, in_channel, out_channel, kernel_size):
        super().__init__()

        self.height, self.width = image_shape
        self.out_channel = out_channel
        self.convgru_cell = ConvGRUCell(in_channel, out_channel, kernel_size)

    def _get_init_state(self, batch_size, dtype):
        state = Variable(torch.zeros(batch_size, self.out_channel, self.height, self.width)).type(dtype)

        return state

    def forward(self, x_sequence, init_hidden=None):
        """
        Args:
            x_sequence shape -> (batch_size, time, c, width, height)
        Return:
            outputs shape -> (time, batch_size, c, width, height)
        """
        seq_len = x_sequence.shape[1]
        dtype = x_sequence.type()
        if init_hidden is None:
            hidden_state = self._get_init_state(x_sequence.shape[0], dtype)
        else:
            hidden_state = init_hidden

        out_list = []
        for t in range(seq_len):
            out, hidden_state = self.convgru_cell(x_sequence[:, t, :, :, :], hidden_state)
            out_list.append(out)

        outputs = torch.stack(out_list, dim=0)

        
        return outputs

