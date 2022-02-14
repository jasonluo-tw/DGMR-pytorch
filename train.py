import torch
import torch.nn as nn
import pytorch_lightning as pl
from DGMR_lightning import DGMR_model

## The model size is the same as paper, if you set base_channels = 24, and pred_step = 18
model = DGMR_model(base_channels=12, pred_step=4)

trainer = pl.Trainer(gpus=1)
trainer.fit(model)

