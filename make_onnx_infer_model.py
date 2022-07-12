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

## load model
model = DGMR_model(in_shape=(384, 384), base_channels=16, pred_step=12, down_step=4, grid_lambda=20)
model = model.load_from_checkpoint('model_weight/v9_only_mse_cwb/DGMR-epoch=084-val_loss=7.4033.ckpt', in_shape=(384, 384), base_channels=16, pred_step=12, down_step=4)

input_sample = torch.randn((1, 4, 1, 384, 384))

#filepath = "DGMR_v9_ep30.onnx"
#model.to_onnx(filepath, input_sample, export_params=True)

## pytorch script pt
#script = model.to_torchscript()
#torch.jit.save(script, "model.pt")

## torch to onnx
#model.generator.eval()
#torch.onnx.export(model.generator,         # model being run 
#         input_sample,       # model input (or a tuple for multiple inputs) 
#         "DGMR_v9_ep30.onnx",       # where to save the model  
#         export_params=True,  # store the trained parameter weights inside the model file 
#         opset_version=10,    # the ONNX version to export the model to 
#         do_constant_folding=True,  # whether to execute constant folding for optimization 
#         input_names = ['modelInput'],   # the model's input names 
#         output_names = ['modelOutput'], # the model's output names 
#         dynamic_axes={'modelInput' : {0 : 'batch_size'},    # variable length axes 
#                                'modelOutput' : {0 : 'batch_size'}}) 


## torch to state_dict
torch.save(model.generator.state_dict(), "state_dict_model/model_v9_TW_mse_ep84.pt")
