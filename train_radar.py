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

def main(args):
    """
    training for DGMR
    """

    input_shape = tuple(map(lambda x: int(x), args.input_shape.split(',')))

    ## The model size is the same as paper, if you set base_channels = 24, and pred_step = 18, down_step=4
    model = DGMR_model(
        train_path = args.train_path,
        valid_path = args.valid_path,
        pic_path   = args.pic_path,
        in_shape = input_shape,
        in_channels = args.in_channels,
        base_channels = args.hidden_base,
        down_step = 4,
        pred_step = args.pred_step,
        use_cuda = True,
        grid_lambda = args.grid_lambda,
        batch_size = args.batch_size,
        gen_sample_nums = args.gen_sample_nums,
    )

    #model = DGMR_model(in_shape=(256, 256), base_channels=16, pred_step=8, down_step=4, grid_lambda=50, batch_size=4)
    
    ## callback function
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        save_top_k=3,
        mode="min",
        dirpath=args.model_dir,
        filename="DGMR-{epoch:03d}-{val_loss:.4f}",
        every_n_epochs=1
    )
    ## EarlyStopping
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.10, patience=10, verbose=False, mode="max")
    
    trainer = pl.Trainer(gpus=args.num_gpus, max_epochs=args.epochs, enable_progress_bar=False, callbacks=[checkpoint_callback, early_stop_callback], log_every_n_steps=4)

    #trainer = pl.Trainer(gpus=1, max_steps=10, enable_progress_bar=False, log_every_n_steps=4)
    #trainer = pl.Trainer(gpus=1, overfit_batches=1, enable_progress_bar=False)
   
    if args.pretrain_model != 'None':
        trainer.fit(model, ckpt_path=args.pretrain_model)
    else:
        trainer.fit(model)
   
    trainer.save_checkpoint(args.model_dir+"/final_model.ckpt")

def argv_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-s', "--input-shape", type=str, default="256,256", help="image_size"
    )
    
    parser.add_argument(
        "-in_c", "--in-channels", type=int, default=1, help="input_channel"
    )
    
    parser.add_argument(
        "-n", "--hidden-base", type=int, default=24, help="hidden size of model"
    )
    
    parser.add_argument(
        "--pred-step", type=int, default=8, help="forecast steps"
    )
    
    parser.add_argument(
        "--grid-lambda", type=int, default=20, help="weight of grid regularization"
    )

    parser.add_argument(
        "--g-lr", type=float, default=5e-5, help="learning rate for generator"
    )
    
    parser.add_argument(
        "--d-lr", type=float, default=2e-4, help="learning rate for discriminator"
    )

    parser.add_argument(
        "--batch-size", type=int, default=16, help="batch size for training"
    )
    
    parser.add_argument(
        "--epochs", type=int, default=80, help="batch size for training"
    )

    parser.add_argument(
        "--gen-sample-nums", type=int, default=6, help="samples for monte Carlo estimation"
    )

    parser.add_argument(
        "--dis-train-step", type=int, default=2, help="discriminator per training step"
    )
    
    parser.add_argument(
        "--pretrain-model", type=str, default='None', help="pretrain model location"
    )
    

    ## container environment
    parser.add_argument("--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"]))
    parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--train-path", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--valid-path", type=str, default=os.environ["SM_CHANNEL_VALID"])
    parser.add_argument("--num-gpus", type=int, default=os.environ["SM_NUM_GPUS"])
    parser.add_argument("--pic-path", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
    
    return parser

if __name__ == '__main__':
    parser = argv_parser()

    main(parser.parse_args())
