import torch
from torch.utils.data import Dataset

import numpy as np

class flowers(Dataset):
    def __init__(self, xPath, yPath, transform=None):
        """
        Args:
            xPath (string): path of the 'trainX.npy' or 'testX.npy' file
            yPath (string): path of the 'trainY.npy' or 'testY.npy' file
            transform (callable, optional): Optional transform to be applied on a sample.
        """

        self.imgs = np.load(xPath, allow_pickle=True)
        self.labels = np.load(yPath, allow_pickle=True)

        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = self.imgs[idx]
        label = self.labels[idx]

        if self.transform:
            img = transform(img)

        #      image label     
        return (img, label )
