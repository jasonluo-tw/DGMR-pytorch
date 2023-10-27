import json
import boto3
import sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker import get_execution_role

import os
os.environ["NCCL_DEBUG"] = "INFO"
#os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

account_id = "662524366798"
role_name = "SRAD-ML"

#role = get_execution_role()
role = f"arn:aws:iam::{account_id}:role/service-role/{role_name}"

bucket = 'dev-radar-nowcasting'
sub_train_folder = 'JP_training/train_data'
#sub_valid_folder = 'JP_training/valid_data'
sub_valid_folder = 'JP_training/small_valid_data'
#sub_valid_folder = 'sage_train_demo'

## TEST
#sub_train_folder = 'sage_train_demo'
#sub_valid_folder = 'sage_train_demo'

## train path, valid path
train_path = f's3://{bucket}/{sub_train_folder}'
valid_path = f's3://{bucket}/{sub_valid_folder}'

conn = boto3.client('s3')
contents = conn.list_objects(Bucket=bucket, Prefix=sub_train_folder)['Contents']

## SageMaker training
estimator = PyTorch(
    entry_point='train_sagemaker.py',
    source_dir='DGMR_sagemaker',
    role=role,
    py_version='py38',
    framework_version='1.10',
    instance_count=1,
    instance_type='ml.g4dn.xlarge', ## Default instance type -> ml.g4dn.xlarge, ml.g5.12xlarge, ml.g4dn.12xlarge
    #instance_type='ml.g4dn.12xlarge',
    hyperparameters={
        'epochs': 50,
        'hidden-base': 12,
        'batch-size': 10,
        'gen-sample-nums': 3,
        'num-workers': 18,
        'dis-train-step': 1,
        'grid-lambda': 40,
        'warmup-iter': 400,
        #'pretrain-model': f's3://{bucket}/JP_training/models/DGMR-epoch=024-val_loss=19.9789.ckpt'
    },
    tags=[{'Key': 'Cost', 'Value': 'dlnowcast'}],
    base_job_name='dlnowcast-train-v10',
    max_run=86400 * 2, ## default: 86400 seconds
    input_mode='FastFile',

    checkpoint_s3_uri=f's3://{bucket}/JP_training/checkpoints',
    image_uri="763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:1.10-gpu-py38",
    #use_spot_instances=True,
    #max_wait=86400 * 5

)

print(estimator.training_image_uri())
estimator.fit({'train': train_path, 'valid': valid_path}, wait=False)
