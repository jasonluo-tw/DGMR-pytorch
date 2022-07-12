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
                                   down_step, prev_step, pred_step, batch_size, use_cuda)

        self.discriminator = Discriminator(in_channels)
        ### set up loss
        self.g_loss   = [0, 0]
        self.d_loss   = [0, 0]
        self.reg_loss = [0, 0]

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
        radar_set = RadarDataSet(folder='../data/training_data/training/train_data')
        self.radar_set = radar_set
        ## valid 
        radar_valid = RadarDataSet(folder='../data/training_data/training/valid_data')
        self.radar_valid = radar_valid

    def setup(self, stage):
        ## Set up train & val dataset
        """
        train_size = int(len(self.radar_set) * 0.8)
        valid_size = len(self.radar_set) - train_size
        self.train_set, self.valid_set = random_split(self.radar_set, 
                [train_size, valid_size],
                generator=torch.Generator().manual_seed(42))
        """
        self.train_set = self.radar_set
        self.valid_set = self.radar_valid

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
        ##
        #X = X * 64
        #Y = Y * 64

        self.global_iter += 1
        g_opt, d_opt = self.optimizers()
        #### warm up for generator
        warmup_epoch = 1
        ####
        discriminator_loss = 999.
        #### optimize discriminator ####
        ## Here train the discriminator 1 times, generator 1 time
        ## set: 4000
        ## current epoch starts from 0
        ##TODO: close discriminator
        """
        if self.current_epoch > warmup_epoch and discriminator_loss >= 0.01:
            for _ in range(1):
                preds = self.generator(X) ## generate predictions
                ####
                fake_score = self.discriminator(X, preds)
                real_score = self.discriminator(X, Y)
                ####
                discriminator_loss = hinge_loss_dis(fake_score, real_score)
                d_opt.zero_grad()
                self.manual_backward(discriminator_loss)
                d_opt.step()
        """

        #### optimize generator ####
        for _ in range(1):
            ## mae weighted
            #preds = self.generator(X)
            preds = [self.generator(X) for _ in range(3)]
            grid_cell_reg = grid_cell_regularizer(torch.stack(preds, dim=0), Y)
            ## mse loss
            #mseLoss = mse_loss(preds*64, Y*64)

            ### concat along time dimension
            #fake_score = self.discriminator(X, preds)
            #hinge_loss = hinge_loss_gen(fake_score)
           
            ##TODO: close discriminator
            #fake_score = [self.discriminator(X, pp) for pp in preds]
            #hinge_loss = hinge_loss_gen(torch.cat(fake_score, dim=0))

            ## deal with grid lambda
            if self.current_epoch > warmup_epoch:
                cc = self.grid_lambda
                ##TODO
                #generator_loss = hinge_loss + cc * grid_cell_reg
                generator_loss = grid_cell_reg
            else:
                cc = -999.
                hinge_loss = -999.
                generator_loss = grid_cell_reg

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
                #'hinge_loss': hinge_loss,
                'grid_reg': grid_cell_reg,
                #'dis_loss': discriminator_loss,
                #'mse_loss': mseLoss
            },
            prog_bar = True
        )
        if self.current_epoch > warmup_epoch:
            ### g_loss
            #self.g_loss[0] += hinge_loss.cpu().detach().numpy()
            #self.g_loss[1] += 1
            ### d_loss
            #self.d_loss[0] += discriminator_loss.cpu().detach().numpy()
            #self.d_loss[1] += 1
            ## reg_loss
            self.reg_loss[0] += grid_cell_reg.cpu().detach().numpy()
            self.reg_loss[1] += 1

        if self.global_iter % 500 == 0:
            ####
            #print(f'epoch:{self.current_epoch},global_step:{self.global_iter},hinge_loss:{hinge_loss},grid_reg:{grid_cell_reg:4f},lambda:{cc:.2f},dis_loss:{discriminator_loss}', flush=True)
            print(f'epoch:{self.current_epoch},global_step:{self.global_iter},grid_cell_reg:{grid_cell_reg}')


    def training_epoch_end(self, all_loss):
        g_loss   = self.g_loss[0] / (self.g_loss[1]+1e-10)
        d_loss   = self.d_loss[0] / (self.d_loss[1]+1e-10)
        reg_loss = self.reg_loss[0] / (self.reg_loss[1]+1e-10)

        print('#### Train Epoch Average result #####', flush=True)
        print(f'epoch:{self.current_epoch}', flush=True)
        #print(f'mean g_loss:{g_loss:.3f}', flush=True)
        #print(f'mean d_loss:{d_loss:.3f}', flush=True)
        print(f'mean reg_loss:{reg_loss:.3f}', flush=True)
        print('#### Train Epoch End #####', flush=True)
        
        self.g_loss   = [0, 0] 
        self.d_loss   = [0, 0]
        self.reg_loss = [0, 0]

    ## Validation set
    def validation_step(self, batch, batch_idx):
        X, Y = batch
        ###
        if self.valid_plot_flag:
            xx = X[0:5]
            yy = Y[0:5]
            preds = self.generator(xx)
            for i in range(xx.shape[0]):
                self.make_visual(xx[i:(i+1)], yy[i:(i+1)], preds[i:(i+1)], i)

            self.valid_plot_flag = False
        
            del xx, yy, preds

        ## calculate error metrics
        Y_hat = self.generator(X)
        ### 
        X = X * 64
        Y = Y * 64
        Y_hat = Y_hat * 64

        ## calculate mse
        loss = F.mse_loss(Y_hat, Y)
        self.log("val_loss", loss, prog_bar=True)

        ## calculate CSI 0.5
        Y_cpu = Y.cpu().detach().numpy()
        Y_hat_cpu = Y_hat.cpu().detach().numpy()

        ts = get_CSI_along_time(Y_cpu[:, :, 0, :, :], Y_hat_cpu[:, :, 0, :, :], threshold=1)
        ts4 = get_CSI_along_time(Y_cpu[:, :, 0, :, :], Y_hat_cpu[:, :, 0, :, :], threshold=4.0)
        del Y_hat

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
        #opt_g = torch.optim.Adam(self.generator.parameters(), lr=5e-5, betas=(0.0, 0.999))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=2e-4, betas=(0.0, 0.999))
        ##
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=1e-4)
        #opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=1e-4)


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

        name = ['pics_new_reg/real', 'pics_new_reg/fake']
        
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


