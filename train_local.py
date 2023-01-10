import argparse
from datetime import timedelta
import json, os, sys

## torch
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

## model script
from DGMR_lightning_radar import DGMR_model

from pytorch_lightning.utilities.rank_zero import rank_zero_only

## The model size is the same as paper, if you set base_channels = 24, and pred_step = 18, down_step=4
model = DGMR_model(
    train_path = "../data/training_data/training/train_data",
    valid_path = "../data/training_data/training/sage_train_demo",
    pic_path = "./train_log/pics/v11",
    in_shape = (256, 256),
    in_channels = 1,
    base_channels = 12,
    down_step = 4,
    pred_step = 12,
    use_cuda = True,
    grid_lambda = 20,
    batch_size = 6,
    gen_sample_nums = 3,
    train_nums = 50000,
    dis_train_step = 2,
    gen_train_step = 1,
    warmup_iter = 200
)

## callback function
checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    save_top_k=5,
    mode="min",
    dirpath='model_weight/v11',
    filename="DGMR-{epoch:03d}-{val_loss:.4f}",
    every_n_epochs=1,
)

#checkpoint2 = ModelCheckpoint(
#    #every_n_train_steps=10,
#    monitor="grid_reg",
#    save_top_k=3,
#    mode="min",
#    #train_time_interval=timedelta(minutes=90),
#    train_time_interval=timedelta(seconds=10),
#    dirpath="model_weight/v9_SageMaker3",
#    filename="DGMR-{epoch:03d}-{step}-{grid_reg:.3f}"
#)

## EarlyStopping
early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=20, verbose=False, mode="min")

trainer = pl.Trainer(gpus=1, 
                     max_epochs=60,
                     #max_steps=10,
                     enable_progress_bar=False, 
                     callbacks=[checkpoint_callback], 
                     log_every_n_steps=4,
                     #overfit_batches=1
                     )

trainer.fit(model,
            #ckpt_path="model_weight/v10_SageMaker/final_model.ckpt",
            )
