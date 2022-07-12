import torch
import torch.nn as nn
import torch.nn.functional as F
from DGMR.model import Generator, Discriminator
from loss import hinge_loss_dis, hinge_loss_gen, grid_cell_regularizer
##TODO: test
from loss import hinge_loss_dis_test
from loss import mse_loss
import pytorch_lightning as pl
## data
from torch.utils.data import random_split, DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt

## add test moving mnist data
import sys
sys.path.append('/home/luo-j/subs')
from data_load import RadarDataSet
from metrics import get_CSI_along_time

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
            batch_size: int = 16,
            **kwargs
    ):
        super().__init__()
        self.global_iter = 0
        self.grid_lambda = grid_lambda
        self.batch_size = batch_size
        self.generator = Generator(in_channels, in_shape, base_channels,
                                   down_step, prev_step, pred_step, use_cuda)

        self.discriminator = Discriminator(in_channels)
        ###
        #self.g_lr = 5e-5

        self.valid_plot_flag = True
        
        ## Important: This property activates manual optimization.
        self.automatic_optimization = False
        #torch.autograd.set_detect_anomaly(True)

    #################
    ####### Test code
    #################
    ##TODO: Use random numbers to test if model can be trained and run
    ## X dims -> (batch, prev_step, channels, width, height)
    ## Y dims -> (batch, pred_step, channels, width, height)
    def prepare_data(self):
        radar_set = RadarDataSet(folder='../data/training_data/training/all_training_test')
        self.radar_set = radar_set

    def setup(self, stage):
        ## Set up train & val dataset
        train_size = int(len(self.radar_set) * 0.9)
        valid_size = len(self.radar_set) - train_size
        self.train_set, self.valid_set = random_split(self.radar_set, 
                [train_size, valid_size],
                generator=torch.Generator().manual_seed(42))

        print('Training set size:', len(self.train_set))
        print('Validation set size:', len(self.valid_set))

    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(
                 dataset=self.train_set,
                 batch_size=self.batch_size,
                 num_workers=4,
                 shuffle=True)

        return train_loader

    def val_dataloader(self):
        val_loader = torch.utils.data.DataLoader(
                 dataset=self.valid_set,
                 batch_size=self.batch_size,
                 num_workers=4,
                 shuffle=False)

        return val_loader

    #################
    ## Test code ends
    #################

    def forward(self, x):
        y = self.generator(x)
        return y

    def training_step(self, batch, batch_idx):
        ## X,Y dims -> (batch, depth, channel, height, width)
        X, Y = batch

        self.global_iter += 1
        g_opt, d_opt = self.optimizers()
        #g_opt = self.optimizers()
        real_seq = torch.cat([X, Y], dim=1)
        #### warm up for generator
        warmup_epoch = 3
        ####
        discriminator_loss = 999.
        #### optimize discriminator ####
        ## Here train the discriminator 1 times, generator 1 time
        ## set: 4000
        ## current epoch starts from 0
        if self.current_epoch > warmup_epoch and discriminator_loss >= 0.01:
            for _ in range(1):
                preds = self.generator(X) ## generate predictions
                ####
                fake_seq = torch.cat([X, preds], dim=1)

                ## concat data
                concat_seq = torch.cat([real_seq, fake_seq], dim=0)

                ## permutation shuffle idx
                shuffle_idx = torch.randperm(concat_seq.shape[0])
                ## shuffle data
                concat_seq = concat_seq[shuffle_idx, :, :, :, :]
                
                concat_scores = self.discriminator(concat_seq, idx=4)
                #real_score, fake_score = torch.chunk(concat_scores, 2, dim=0)
            
                ## make targets
                real_ones = -torch.ones([real_seq.shape[0], 2, 1]).type_as(real_seq)
                fake_ones = torch.ones([fake_seq.shape[0], 2, 1]).type_as(fake_seq)
                concat_ones = torch.cat([real_ones, fake_ones], dim=0)
                ## shuffle ones
                concat_ones = concat_ones[shuffle_idx, :, :]

                discriminator_loss = hinge_loss_dis_test(concat_scores, concat_ones)
                d_opt.zero_grad()
                self.manual_backward(discriminator_loss)
                d_opt.step()

        #### optimize generator ####
        for _ in range(1):
            #preds = [self.generator(X) for _ in range(2)]
            #grid_cell_reg = grid_cell_regularizer(torch.stack(preds, dim=0), Y)
            preds = self.generator(X)
            mseLoss = mse_loss(preds*64, Y*64)
            ## concat along time dimension
            gen_sequences = torch.cat([X, preds], dim=1)
            dis_scores = self.discriminator(gen_sequences, idx=4)
            hinge_loss = hinge_loss_gen(dis_scores)
            #gen_sequences = [torch.cat([X, g], dim=1) for g in preds]
            #disc_scores = [self.discriminator(f, idx=4) for f in gen_sequences]
            #hinge_loss = hinge_loss_gen(torch.cat(disc_scores, dim=0))

            ## deal with grid lambda
            if self.current_epoch > warmup_epoch:
                cc = self.grid_lambda
                #generator_loss = hinge_loss + cc * grid_cell_reg
                generator_loss = hinge_loss + cc * mseLoss
            else:
                cc = -999.
                hinge_loss = -999.
                generator_loss = mseLoss
            #####
            ###TODO: only MSELoss
            #preds = self.generator(X)
            #mseLoss = mse_loss(preds*64, Y*64)
            #generator_loss = mseLoss 

            ### Warm up schedule for opt.
            #if self.global_iter < 500:
            #    lr_scale = min(1., float(self.global_iter+1) / 500.)
            #    for pg in g_opt.param_groups:
            #        pg['lr'] = lr_scale * self.g_lr

            g_opt.zero_grad()
            self.manual_backward(generator_loss)
            g_opt.step()

        ##TODO: store loss, and visualization part
        self.log_dict(
            {
                'hinge_loss': hinge_loss,
                #'grid_reg': grid_cell_reg,
                'dis_loss': discriminator_loss,
                'mse_loss': mseLoss
            },
            prog_bar = True
        )
        ### visualziae step
        #if self.global_iter % 500 == 0:
        #    xx = X[0:1]
        #    yy = Y[0:1]
        #    preds = self.generator(xx)
        #    self.make_visual(xx, yy, preds)

        if self.global_iter % 100 == 0:
            ####
            print(f'epoch:{self.current_epoch},global_step:{self.global_iter},hinge_loss:{hinge_loss},mse_loss:{mseLoss:4f},lambda:{cc:.2f},dis_loss:{discriminator_loss}', flush=True)
            #print(f'epoch:{self.current_epoch},global_step:{self.global_iter},mseloss:{mseLoss}')

    ## Validation set
    def validation_step(self, batch, batch_idx):
        X, Y = batch
        Y_hat = self.generator(X)

        ## calculate mse
        loss = F.mse_loss(Y_hat*64, Y*64)
        self.log("val_loss", loss, prog_bar=True)

        ## calculate CSI 0.5
        Y_cpu = Y.cpu().detach().numpy()
        Y_hat_cpu = Y_hat.cpu().detach().numpy()

        ts = get_CSI_along_time(Y_cpu[:, :, 0, :, :]*64, Y_hat_cpu[:, :, 0, :, :]*64, threshold=1)
        ts4 = get_CSI_along_time(Y_cpu[:, :, 0, :, :]*64, Y_hat_cpu[:, :, 0, :, :]*64, threshold=4.0)
        del Y_hat

        if self.valid_plot_flag:
            xx = X[0:5]
            yy = Y[0:5]
            del X, Y
            preds = self.generator(xx)
            for i in range(xx.shape[0]):
                self.make_visual(xx[i:(i+1)], yy[i:(i+1)], preds[i:(i+1)], i)

            self.valid_plot_flag = False
        
            del xx, yy, preds

        return loss, ts, ts4

    def validation_epoch_end(self, all_loss):
        mse = [i[0] for i in all_loss]
        ts  = [i[1] for i in all_loss]
        ts4  = [i[2] for i in all_loss]
        mse = torch.mean(torch.stack(mse))

        ts  = np.mean(ts, axis=0)
        ts_out = ','.join([f'{i:.3f}' for i in ts])
        
        ts4  = np.mean(ts4, axis=0)
        ts4_out = ','.join([f'{i:.3f}' for i in ts4])

        print('#### Validation #####', flush=True)
        print(f'epoch:{self.current_epoch}', flush=True)
        print(f'mse:{mse:.3f}', flush=True)
        print(f'CSI-1:{ts_out}', flush=True)
        print(f'CSI-4:{ts4_out}', flush=True)
        print('#### Validation end ####', flush=True)

        self.valid_plot_flag = True

    def configure_optimizers(self):
        #opt_g = torch.optim.Adam(self.generator.parameters(), lr=1e-5, betas=(0.0, 0.999))
        #opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=1e-4, betas=(0.0, 0.999))
        ##
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=1e-4)
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=1e-4)


        return opt_g, opt_d
        #return opt_g

    def make_visual(self, xx, yy, preds, fig_index=None):
        real = torch.cat([xx, yy], dim=1).cpu().detach() * 64
        fake = torch.cat([xx, preds], dim=1).cpu().detach() * 64

        real = np.squeeze(real.numpy()) 
        ##
        fake = np.squeeze(fake.numpy())
        fake = np.where(fake > 64, 64, fake)
        fake = np.where(fake < 0, 0, fake)

        name = ['pics/real', 'pics/fake']
        
        for index, pics in enumerate([real, fake]):
            plt.figure(figsize=(12, 10))
            for t in range(pics.shape[0]):
                plt.subplot(3, 4, t+1)
                plt.contourf(pics[t], levels=np.arange(0, 65, 4), cmap='rainbow')
                plt.title(t)
                plt.axis('off')

            folder = name[index]
            if fig_index is None:
                plt.savefig(f'{folder}/{self.current_epoch}.png', bbox_inches='tight')
            else:
                plt.savefig(f'{folder}/{self.current_epoch}_{fig_index}.png', bbox_inches='tight')
            plt.close()


