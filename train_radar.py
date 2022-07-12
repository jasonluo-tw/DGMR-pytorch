import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

##TODO
from DGMR_lightning_radar_TW import DGMR_model
#from DGMR_lightning_radar2 import DGMR_model

## The model size is the same as paper, if you set base_channels = 24, and pred_step = 18, down_step=4
## TW
model = DGMR_model(in_shape=(384, 384), base_channels=16, pred_step=12, down_step=4, grid_lambda=200, batch_size=4)
#model = DGMR_model(in_shape=(256, 256), base_channels=16, pred_step=8, down_step=4, grid_lambda=50, batch_size=4)

## callback function
checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    save_top_k=3,
    mode="min",
    dirpath="model_weight/v9_only_mse_cwb",
    filename="DGMR-{epoch:03d}-{val_loss:.4f}",
    every_n_epochs=1
)
## EarlyStopping
early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=10, verbose=False, mode="max")

trainer = pl.Trainer(gpus=1, max_epochs=90, enable_progress_bar=False, callbacks=[checkpoint_callback], log_every_n_steps=4)
#trainer = pl.Trainer(gpus=1, max_steps=10, enable_progress_bar=False, log_every_n_steps=4)
#trainer = pl.Trainer(gpus=1, overfit_batches=1, enable_progress_bar=False)

trainer.fit(model, ckpt_path='model_weight/v9_only_mse/final_model_radar_v9_only_mse.ckpt')
#trainer.fit(model)

trainer.save_checkpoint("model_weight/final_model_radar_v9_only_mse_cwb.ckpt")
