import torch
import numpy as np
import os

class RadarDataSet(torch.utils.data.Dataset):
    def __init__(self, folder):
        self.data_files = os.listdir(folder)
        self.folder = folder

    def __getitem__(self, idx):
        data = np.load(os.path.join(self.folder, self.data_files[idx]))
        data = np.nan_to_num(data, nan=0.0)
        data = data.astype('float32')
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


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    dset = RadarDataSet(folder='../data/training_data/training/all_training')
    loader = torch.utils.data.DataLoader(dset, batch_size=32, num_workers=2, shuffle=True)

    for i, j in loader:
        kk = np.squeeze(np.concatenate((i[0], j[0]), axis=0))
        for index, dd in enumerate(kk):
            plt.subplot(3, 4, index+1)
            plt.contourf(dd, levels=np.arange(0, 65, 2), cmap='rainbow')
            
        plt.savefig('test.png', bbox_inches='tight')

        break


