import argparse
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import json, os

## model script
#from DGMR_lightning_radar_TW import DGMR_model
from DGMR_lightning_radar2 import DGMR_model


## The model size is the same as paper, if you set base_channels = 24, and pred_step = 18, down_step=4
model = DGMR_model(
    train_path = "../data/training_data/training/valid_data",
    valid_path = None,
    pic_path   = "./pics/",
    in_shape = (404, 404),
    in_channels = 1,
    base_channels = 16,
    down_step = 4,
    pred_step = 12,
    use_cuda = True,
    grid_lambda = 50,
    batch_size = 4,
    gen_sample_nums = 1,
)

## callback function
checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    save_top_k=3,
    mode="min",
    dirpath='model_weights/test',
    filename="DGMR-{epoch:03d}-{val_loss:.4f}",
    every_n_epochs=1
)
## EarlyStopping
early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.10, patience=10, verbose=False, mode="max")

trainer = pl.Trainer(gpus=1, max_epochs=1, enable_progress_bar=False, callbacks=[checkpoint_callback, early_stop_callback], log_every_n_steps=4)

#trainer = pl.Trainer(gpus=1, max_steps=10, enable_progress_bar=False, log_every_n_steps=4)
#trainer = pl.Trainer(gpus=1, overfit_batches=1, enable_progress_bar=False)

#trainer.fit(model, ckpt_path=args.pretrain_model)
trainer.fit(model)
   
