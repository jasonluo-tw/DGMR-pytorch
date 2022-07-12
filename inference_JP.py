import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import numpy as np
## plot
import matplotlib.pyplot as plt
import matplotlib.animation as animation

###
from DGMR_lightning_radar2 import DGMR_model
###
import sys
from data_load import RadarDataSet

sys.path.append('/home/luo-j/subs')
from plot2 import PLOT
from metrics import get_CSI_along_time
from read import read_radar_data, dBZ2rain

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
## load model
model = DGMR_model(in_shape=(256, 256), base_channels=16, pred_step=8, down_step=4, grid_lambda=20)
model = model.load_from_checkpoint('model_weight/v9_new_data/DGMR-epoch=045-val_loss=10.8393.ckpt', in_shape=(256, 256), base_channels=16, pred_step=8, down_step=4)
model.to(device)

## load data
root_folder = '../data/training_data/training/valid_data'
radar_set = RadarDataSet(folder=root_folder)
file_list = radar_set.data_files

## get validation set
valid_set = radar_set
valid_size = len(valid_set)

## get loader
val_loader = DataLoader(
    dataset=valid_set,
    batch_size=32,
    num_workers=1,
    shuffle=False
)

steps = valid_size // 32 + 1
print('total steps:', steps)

all_csi = {1: [], 4: [], 8: []}
all_mse = []
num = 0
for seq, seq_target in val_loader:
    with torch.no_grad():
        seq_preds = torch.quantile(torch.stack([model(seq) for i in range(10)], dim=0), 0.5, dim=0)
        #seq_preds = model(seq.to(device))

    seq = np.squeeze(seq.numpy() * 64)
    seq_target = np.squeeze(seq_target.detach().numpy() * 64)
    seq_preds = np.squeeze(seq_preds.cpu().detach().numpy() * 64)

    #seq_preds = np.where(seq_preds < 0.2, np.nan, seq_preds)
    #seq_target = np.where(seq_target < 0.2, np.nan, seq_target)
    # dim -> (samples, frames, channel, width, height)
    ####
    """
    print(file_list[0])
    splits= file_list[0].split('_')
    date = splits[0]
    domain = splits[1][1:]
    ####
    ## plot
    pp = PLOT(seq, seq_preds, seq_target)
    pp.plot_animation(prefix=f'./pics/ground_truth{num}/', thres=1.0)

    num += 1
    if num == 2:
        break
    """

    for thres in all_csi:
        all_ts = get_CSI_along_time(seq_preds, seq_target, thres)
        all_csi[thres].append(all_ts)

    all_mse.append(np.mean( (seq_preds-seq_target)**2 ))

    num += 1
    print(f'{num}/{steps}')

for thres in all_csi:
    ts = np.mean(all_csi[thres], axis=0)
    print(f'CSI-{thres}:{ts}')

mse = np.mean(all_mse)
print(f'mse:{mse}')
