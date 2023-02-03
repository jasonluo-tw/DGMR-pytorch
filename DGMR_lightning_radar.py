## torch
import torch, sys
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader, TensorDataset
import pytorch_lightning as pl
## DGMR
from DGMR.model import Generator, Discriminator
from DGMR.loss import hinge_loss_dis, hinge_loss_gen, grid_cell_regularizer
from DGMR.loss import hinge_loss_dis_test
from DGMR.loss import mse_loss
from DGMR.metrics import get_CSI_along_time
from DGMR.data_load import RadarDataSet

import numpy as np
import matplotlib.pyplot as plt
import logging, os

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

#logging.basicConfig(
#    filename=os.environ["SM_OUTPUT_DATA_DIR"]+'/train.log',
#    level=logging.INFO,
#    #format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#    format="",
#    datefmt='%Y-%m-%d %H:%M:%S'
#)

class DGMR_model(pl.LightningModule):
    """
    Deep Generative Model of Radar
    """
    def __init__(
            self,
            train_path=None,
            valid_path=None,
            pic_path=None, 
            in_shape: list = (256, 256),
            in_channels: int = 1,
            base_channels: int = 24,
            down_step: int = 4,
            prev_step: int = 4,
            pred_step: int = 18,
            use_cuda: bool = True,
            grid_lambda: float = 20,
            batch_size: int = 16,
            warmup_iter: int = 0,
            gen_train_step: int = 1,
            dis_train_step: int = 1,
            gen_sample_nums: int = 6,
            train_gen_only: bool = False,
            opt_g_lr: float = 5e-5,
            opt_d_lr: float = 1e-4,
            data_num_workers: int = 4,
            train_nums = None,
            gpu_nums: int = 1,
            **kwargs
    ):
        super().__init__()
        self.global_iter = 0
        self.grid_lambda = grid_lambda
        self.batch_size = batch_size
        self.gpu_nums = gpu_nums
        self.train_nums = train_nums

        self.data_num_workers = data_num_workers
        self.warmup_iter = warmup_iter
        self.gen_train_step = gen_train_step
        self.dis_train_step = dis_train_step
        self.gen_sample_nums = gen_sample_nums
        ## path
        self.train_path = train_path
        self.valid_path = valid_path
        self.pic_path = pic_path
        ## optimizer 
        self.opt_g_lr = opt_g_lr
        self.opt_d_lr = opt_d_lr
        ##TODO: pred step
        self.pred_step = pred_step

        ## bools
        self.train_gen_only = train_gen_only
        
        ## init Generator
        self.generator = Generator(in_channels, in_shape, base_channels,
                                   down_step, prev_step, pred_step, batch_size, use_cuda)

        ## init Discriminator
        if not self.train_gen_only:
            self.discriminator = Discriminator(in_channels, base_channels)

        ### set up loss
        self.g_loss   = [0, 0]
        self.d_loss   = [0, 0]
        self.reg_loss = [0, 0]

        self.valid_plot_flag = True
        
        ## Important: This property activates manual optimization.
        self.automatic_optimization = False
        #torch.autograd.set_detect_anomaly(True)

    def prepare_data(self):
        """
        X dims -> (batch, prev_step, channels, width, height)
        Y dims -> (batch, pred_step, channels, width, height)
        """
        assert self.train_path is not None
        radar_set = RadarDataSet(folder=self.train_path, 
                                 shuffle=True, 
                                 nums=self.train_nums, 
                                 compress_int16=True)

        self.radar_set = radar_set

        ## valid 
        if self.valid_path is not None:
            radar_valid = RadarDataSet(folder=self.valid_path, compress_int16=True)
            self.radar_valid = radar_valid
        else:
            self.radar_valid = None

    def setup(self, stage):
        ## Set up train & val dataset
        if self.valid_path is None:
            train_size = int(len(self.radar_set) * 0.8)
            valid_size = len(self.radar_set) - train_size
            self.train_set, self.valid_set = random_split(self.radar_set, 
                    [train_size, valid_size],
                    generator=torch.Generator().manual_seed(42))
        else:
            self.train_set = self.radar_set
            self.valid_set = self.radar_valid

        if self.global_rank == 0:
            logger.info(f'Training set size: {len(self.train_set)}')
            logger.info(f'Validation set size: {len(self.valid_set)}')

    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(
                 dataset=self.train_set,
                 batch_size=self.batch_size,
                 num_workers=self.data_num_workers,
                 pin_memory=True,
                 shuffle=True)

        return train_loader

    def val_dataloader(self):
        val_loader = torch.utils.data.DataLoader(
                 dataset=self.valid_set,
                 batch_size=self.batch_size,
                 num_workers=self.data_num_workers,
                 pin_memory=True,
                 shuffle=False)

        return val_loader

    def forward(self, x):
        y = self.generator(x)
        return y

    def training_step(self, batch, batch_idx):
        ## X,Y dims -> (batch, depth, channel, height, width)
        X, Y = batch
        ##
        #X = X * 64
        #Y = Y * 64

        #self.global_iter += 1
        g_opt, d_opt = self.optimizers()

        discriminator_loss = 999.

        #### optimize discriminator ####
        ## Here train the discriminator 1 times, generator 1 time
        ## set: 4000
        ## current epoch starts from 0
        if self.global_step >= self.warmup_iter and not self.train_gen_only:
            for _ in range(self.dis_train_step):
                preds = self.generator(X) ## generate predictions

                fake_score = self.discriminator(X, preds)
                real_score = self.discriminator(X, Y)

                discriminator_loss = hinge_loss_dis(fake_score, real_score)
                d_opt.zero_grad()
                self.manual_backward(discriminator_loss)
                d_opt.step()

        #### optimize generator ####
        for _ in range(self.gen_train_step):
            ## mae weighted
            preds = [self.generator(X) for _ in range(self.gen_sample_nums)]
            grid_cell_reg = grid_cell_regularizer(torch.stack(preds, dim=0), Y)

            ##TODO
            ## mse loss
            #mseLoss = mse_loss(preds*64, Y*64)
            ### concat along time dimension
            #fake_score = self.discriminator(X, preds)
            #hinge_loss = hinge_loss_gen(fake_score)
            
            if not self.train_gen_only:
                fake_score = [self.discriminator(X, pp) for pp in preds]
                hinge_loss = hinge_loss_gen(torch.cat(fake_score, dim=0))

            ## deal with grid lambda
            if self.global_step >= self.warmup_iter and not self.train_gen_only:
                cc = self.grid_lambda
                generator_loss = hinge_loss + cc * grid_cell_reg
            else:
                cc = -999. 
                hinge_loss = -999.
                generator_loss = grid_cell_reg

            ##TODO: Warm up schedule for opt.
            #if self.global_step < 500:
            #    lr_scale = min(1., float(self.global_step+1) / 500.)
            #    for pg in g_opt.param_groups:
            #        pg['lr'] = lr_scale * self.g_lr

            g_opt.zero_grad()
            self.manual_backward(generator_loss)
            g_opt.step()

        self.log_dict(
            {
                'hinge_loss': hinge_loss,
                'grid_reg': grid_cell_reg,
                'dis_loss': discriminator_loss,
                #'mse_loss': mseLoss
            },
            prog_bar = True
        )

        if not isinstance(hinge_loss, float):
            ## g_loss
            self.g_loss[0] += hinge_loss.cpu().detach().numpy()
            self.g_loss[1] += 1
            ## d_loss
            self.d_loss[0] += discriminator_loss.cpu().detach().numpy()
            self.d_loss[1] += 1

        ## reg_loss
        self.reg_loss[0] += grid_cell_reg.cpu().detach().numpy()
        self.reg_loss[1] += 1

        if self.global_step % 500 == 0 and self.global_rank == 0:
            logger.info('epoch:{:d},global_step:{:d},hinge_loss:{:.4f},grid_reg:{:.4f},lambda:{:.2f},dis_loss:{:.4f}'.format(
                self.current_epoch,
                self.global_step,
                hinge_loss,
                grid_cell_reg,
                cc,
                discriminator_loss
            ))

    def training_epoch_end(self, all_loss):
        g_loss   = self.g_loss[0] / (self.g_loss[1]+1e-10)
        d_loss   = self.d_loss[0] / (self.d_loss[1]+1e-10)
        reg_loss = self.reg_loss[0] / (self.reg_loss[1]+1e-10)

        if self.global_rank == 0:
            logger.info('#### Train Epoch Average result #####')
            logger.info(f'epoch:{self.current_epoch}')
            logger.info(f'mean g_loss:{g_loss:.3f}')
            logger.info(f'mean d_loss:{d_loss:.3f}')
            logger.info(f'mean reg_loss:{reg_loss:.3f}')
            logger.info('#### Train Epoch End #####')
        
        self.g_loss   = [0, 0] 
        self.d_loss   = [0, 0]
        self.reg_loss = [0, 0]

    def validation_step(self, batch, batch_idx):
        """
        Validation set
        """
        X, Y = batch
        Y_hat = self.generator(X)
        ##  transform
        X = X * 64
        Y = Y * 64
        Y_hat = Y_hat * 64

        ## calculate error metrics
        ## calculate mse
        loss = F.mse_loss(Y_hat, Y)

        ## put data from GPU to CPU
        X_cpu = X.cpu().detach().numpy()
        Y_cpu = Y.cpu().detach().numpy()
        Y_hat_cpu = Y_hat.cpu().detach().numpy()

        ## calculate CSI 1.0, 4.0
        ts_arr = []
        for thres in [1.0, 4.0]:
            ts = get_CSI_along_time(Y_cpu[:, :, 0, :, :], 
                                    Y_hat_cpu[:, :, 0, :, :], 
                                    threshold=thres)
            ts_arr.append(ts)
        
        ## visualization
        if self.global_rank == 0 or self.global_rank == 1: 
            if self.valid_plot_flag:
                xx = X_cpu[:5]
                yy = Y_cpu[:5]
                preds = Y_hat_cpu[:5]
                for i in range(xx.shape[0]):
                    self.make_visual(xx[i:(i+1)], yy[i:(i+1)], preds[i:(i+1)], i)

                self.valid_plot_flag = False
        
                del xx, yy, preds

        del Y_hat

        ## ts_arr[0] -> 1.0, ts_arr[1] -> 4.0
        if self.gpu_nums == 1:
            return [loss, ts_arr[0], ts_arr[1]]  
        else:
            return loss, ts_arr[0], ts_arr[1]  

    def validation_epoch_end(self, all_loss):
        ## for distribution training, gather all loss from different ranks(GPU)
        all_loss_nodes = self.all_gather(all_loss)

        ## calculate error
        if self.global_rank == 0:
            ## mse
            mse = torch.mean(torch.stack([i[0] for i in all_loss_nodes]))
            mse = mse.cpu().detach()#.numpy()

            ## TS
            ## [tensor(1, 2, 3, 4), ..., *12] -> multiple
            ts = [torch.stack(i[1]) for i in all_loss_nodes]
            if self.gpu_nums > 1:
                ts = torch.mean(ts, dim=-1)
            ## average different samples
            ts = torch.mean(torch.stack(ts), dim=0).cpu().detach().numpy()
            ts_out = ','.join([f'{i:.3f}' for i in ts])

            ## TS-4
            ## [tensor(1, 2, 3, 4), ..., *12]
            ts4 = [torch.stack(i[2]) for i in all_loss_nodes]
            if self.gpu_nums > 1:
                ts4 = torch.mean(ts4, dim=-1)
            ## average different samples
            ts4 = torch.mean(torch.stack(ts4), dim=0).cpu().detach().numpy()
            ts4_out = ','.join([f'{i:.3f}' for i in ts4])
            
            ##TODO:self.log can not use the sync_dict=True arguments, 
            ##     or the script will be locked...Not sure why...
            self.log("val_loss", mse, prog_bar=True)
            
            ## logging
            logger.info('#### Validation #####')
            logger.info(f'epoch:{self.current_epoch}')
            logger.info(f'mse:{mse:.3f}')
            logger.info(f'CSI-1:{ts_out}')
            logger.info(f'CSI-4:{ts4_out}')
            logger.info('#### Validation end ####')

        if self.global_rank == 0 or self.global_rank == 1:
            self.valid_plot_flag = True

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.opt_g_lr, betas=(0.0, 0.999))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.opt_d_lr, betas=(0.0, 0.999))

        return opt_g, opt_d

    def make_visual(self, xx, yy, preds, fig_index=None):
        """
        xx, yy, preds -> np.array
        """
        real = np.concatenate([xx, yy], axis=1)
        fake = np.concatenate([xx, preds], axis=1)

        real = np.squeeze(real) 
        ##
        fake = np.squeeze(fake)
        fake = np.where(fake > 64, 64, fake)
        fake = np.where(fake < 0.2, np.nan, fake)
        real = np.where(real < 0.2, np.nan, real)

        ##(Solved) Error occured, when using distributed training. We have to if set global_rank == 0
        ##(Solved) error message -> ValueError: ContourSet must be in current Axes
        name = [self.pic_path+'/real', self.pic_path+'/fake']
        ## check if folder path exists or not
        for folder in name:
            if not os.path.exists(folder):
                os.makedirs(folder, exist_ok=True)
       
        for index, pics in enumerate([real, fake]):
            plt.figure(figsize=(12, 10))
            for t in range(pics.shape[0]):
                if self.pred_step == 8 or self.pred_step == 6:
                    plt.subplot(3, 4, t+1)
                elif self.pred_step == 12:
                    plt.subplot(4, 4, t+1)

                plt.contourf(pics[t], levels=np.arange(0, 65, 4), cmap='rainbow')
                plt.title(t)
                plt.axis('off')

            folder = name[index]
            if fig_index is None:
                plt.savefig(f'{folder}/{self.current_epoch}_rank{self.global_rank}.png', 
                            bbox_inches='tight')
            else:
                plt.savefig(f'{folder}/{self.current_epoch}_{fig_index}_rank{self.global_rank}.png', 
                            bbox_inches='tight')

            plt.close()


