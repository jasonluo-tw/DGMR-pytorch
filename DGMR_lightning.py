import torch
import torch.nn as nn
from DGMR.model import Generator, Discriminator
from loss import hinge_loss_dis, hinge_loss_gen, grid_cell_regularizer
import pytorch_lightning as pl
## data
from torch.utils.data import random_split, DataLoader, TensorDataset
import numpy as np

class DGMR_model(pl.LightningModule):
    """
    Deep Generative Model of Radar
    """
    def __init__(
            self,
            in_shape: list = (256, 256),
            in_channels: int = 1,
            base_channels: int = 24,
            down_step: int = 4,
            prev_step: int = 4,
            pred_step: int = 18,
            use_cuda: bool = True,
            grid_lambda: float = 20,
            **kwargs
    ):
        super().__init__()
        self.global_iter = 0
        self.grid_lambda = 20
        self.generator = Generator(in_channels, in_shape, base_channels,
                                   down_step, prev_step, pred_step, use_cuda)

        self.discriminator = Discriminator(in_channels)
        
        ## Important: This property activates manual optimization.
        self.automatic_optimization = False


    #################
    ####### Test code
    #################
    ##TODO: Use random numbers to test if model can be trained and run
    ## X dims -> (batch, prev_step, channels, width, height)
    ## Y dims -> (batch, pred_step, channels, width, height)
    def prepare_data(self):
        X = np.random.random((200, 4, 1, 256, 256))
        Y = np.random.random((200, 4, 1, 256, 256))
        
        ## convert numpy array to Tensor
        self.x_tensor = torch.from_numpy(X).float()
        self.y_tensor = torch.from_numpy(Y).float()

        training_dataset = TensorDataset(self.x_tensor, self.y_tensor)
        self.training_dataset = training_dataset

    def setup(self, stage):
        data = self.training_dataset
        self.train_data, self.val_data = random_split(data, [160, 40])

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=2, num_workers=4)

    #################
    ## Test code ends
    #################

    def forward(self, x):
        x = self.generator(x)
        return x

    def training_step(self, batch, batch_idx):
        ## X,Y dims -> (batch, depth, channel, height, width)
        X, Y = batch

        self.global_iter += 1
        g_opt, d_opt = self.optimizers()
        ###
        real_seq = torch.cat([X, Y], dim=1)
        #### optimize discriminator ####
        ## Here train the discriminator 2 times, generator 1 time
        for _ in range(2):
            preds = self(X) ## generate predictions
            fake_seq = torch.cat([X, preds], dim=1)
            #### 
            concat_seq = torch.cat([real_seq, fake_seq], dim=0)
            concat_outputs = self.discriminator(concat_seq)

            real_score, fake_score = torch.split(concat_outputs, 1, dim=1)
            discriminator_loss = hinge_loss_dis(fake_score, real_score)
            d_opt.zero_grad()
            self.manual_backward(discriminator_loss)
            d_opt.step()

        #### optimize generator ####
        preds = [self(X) for _ in range(6)]
        grid_cell_reg = grid_cell_regularizer(torch.stack(preds, dim=0), Y)
        ## concat along time dimension
        fake_seq = [torch.cat([X, pp], dim=1) for pp in preds]
        
        gen_scores = []
        for g_seq in fake_seq:
            concat_seq = torch.cat([real_seq, g_seq], dim=0)
            concat_outputs = self.discriminator(concat_seq)
            _, fake_score = torch.split(concat_outputs, 1, dim=1)
            ## append the results
            gen_scores.append(fake_score)

        hinge_loss = hinge_loss_gen(torch.cat(gen_scores, dim=0))
        generator_loss = hinge_loss + self.grid_lambda * grid_cell_reg
        g_opt.zero_grad()
        self.manual_backward(generator_loss)
        g_opt.step()


        ##TODO: store loss, and visualization part


    def configure_optimizers(self):

        opt_g = torch.optim.Adam(self.generator.parameters())
        opt_d = torch.optim.Adam(self.discriminator.parameters())

        return opt_g, opt_d
