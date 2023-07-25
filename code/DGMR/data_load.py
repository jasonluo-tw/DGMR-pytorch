import torch
import numpy as np
import os
import gzip

class RadarDataSet(torch.utils.data.Dataset):
    def __init__(self, folder, data_files=None, nums=None, shuffle=True, seed=20, compress_int16=True):
        ##TODO:
        self.compress_int16 = compress_int16
        np.random.seed(seed)
        if data_files is not None:
            self.data_files = data_files
        else:
            self.data_files = os.listdir(folder)

        if shuffle:
            np.random.shuffle(self.data_files)
        if nums is not None:
            self.data_files = self.data_files[:nums]

        self.folder = folder

    def __getitem__(self, idx):
        f = gzip.open(os.path.join(self.folder, self.data_files[idx]), 'rb')
        if self.compress_int16:
            data = np.load(f).astype('float32') / 100.
        else:
            data = np.load(f).astype('float32')
            data = np.where(data <= 0.01, 0, data)

        data = np.nan_to_num(data, nan=0.0)
        #data = data.astype('float32')
        ## if data >= 64, data = 64
        data = np.where(data > 100, 100, data)
        #data = data / 64.
        ## dims -> (1, 256, 256, 12)
        data = np.moveaxis(data, -1, 0)
        ## dims -> (12, 1, 256, 256)
        X = data[:4]
        Y = data[4:]

        return X, Y

    def __len__(self):
        return len(self.data_files)

