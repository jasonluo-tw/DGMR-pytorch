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
from plot import PLOT
from metrics import get_CSI_along_time
from read import read_radar_data, dBZ2rain

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
## load model
model = DGMR_model(in_shape=(256, 256), base_channels=16, pred_step=12, down_step=4, grid_lambda=20)
model = model.load_from_checkpoint('model_weight/v9_new_data/DGMR-epoch=030-val_loss=10.9956.ckpt', in_shape=(256, 256), base_channels=16, pred_step=12, down_step=4)
model.to(device)

## Read CWB data
root_folder = "../data/CWB"
ZR_type = "TW_Conv_W12"
radar_data, dt = read_radar_data(root_folder, "20220327")
dt = list(map(lambda x: x.strftime('%Y%m%d%H%M'), dt))
## preprocess
radar_data = dBZ2rain(radar_data, ZR_type)

radar_data = np.where(radar_data < 0, 0, radar_data)
radar_data = np.where(radar_data > 64, 64, radar_data)


## normalize
radar_data = radar_data / 64.
radar_data = radar_data.astype('float32')

target_index = dt.index("202203271900")
#cut_x = [176, 560] ## 384x384
#cut_y = [176, 560] ## 384x384

cut_x = [240, 496] ## 256x256
cut_y = [240, 496] ## 256x256
#######
input_time = dt[(target_index-3):target_index+1]
inputx = radar_data[(target_index-3):target_index+1]
inputx = inputx[:, cut_x[0]:cut_x[1], cut_y[0]:cut_y[1]]
inputx = np.expand_dims(inputx, axis=1)
inputx = np.expand_dims(inputx, axis=0)
seq = torch.from_numpy(inputx)
## target
seq_target = radar_data.copy()[(target_index+1):(target_index+1+12)]
seq_target = seq_target[:, cut_x[0]:cut_x[1], cut_y[0]:cut_y[1]] * 64

print(seq_target.shape, seq.shape)

all_csi = {1: [], 4: [], 8: []}
all_mse = []
num = 0
with torch.no_grad():
    #seq_preds = [model(seq.to(device)) for _ in range(3)]
    #seq_preds = torch.mean(torch.stack(seq_preds, dim=0), dim=0)
    #####
    seq_preds = model(seq.to(device))

seq = np.squeeze(seq.numpy() * 64)
seq = np.where(seq < 1.0, np.nan, seq)
#seq_target = np.squeeze(seq_target.detach().numpy() * 64)
seq_preds = np.squeeze(seq_preds.cpu().detach().numpy() * 64)
seq_preds = np.where(seq_preds < 1.0, np.nan, seq_preds)
# dim -> (samples, frames, channel, width, height)
seq = np.expand_dims(seq, axis=0)
seq_preds = np.expand_dims(seq_preds, axis=0)

seq_target = np.expand_dims(seq_target, axis=0)
seq_target = np.where(seq_target < 1.0, np.nan, seq_target)

print(seq.shape, seq_preds.shape, seq_target.shape)
pp = PLOT(seq, seq_preds, seq_target)
pp.plot_animation(prefix=f'./pics/test_CWB/', thres=1.0)

