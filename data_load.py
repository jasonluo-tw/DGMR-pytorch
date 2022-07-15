import torch
import numpy as np
import os
import gzip

class RadarDataSet(torch.utils.data.Dataset):
    def __init__(self, folder):
        ##TODO:
        self.data_files = os.listdir(folder)
        self.folder = folder

    def __getitem__(self, idx):
        f = gzip.open(os.path.join(self.folder, self.data_files[idx]), 'rb')
        data = np.load(f).astype('float32') / 100.
        data = np.nan_to_num(data, nan=0.0)
        #data = data.astype('float32')
        ## if data >= 64, data = 64
        data = np.where(data > 64, 64, data)
        data = data / 64.
        ## dims -> (1, 256, 256, 12)
        data = np.moveaxis(data, -1, 0)
        ## dims -> (12, 1, 256, 256)
        X = data[:4]
        Y = data[4:]

        return X, Y

    def __len__(self):
        return len(self.data_files)

