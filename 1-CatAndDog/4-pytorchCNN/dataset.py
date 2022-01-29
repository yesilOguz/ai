import torch
from torch.utils.data import Dataset

import numpy as np

class CatAndDog(Dataset):
    def __init__(self, img_file, label_file, transform=None):
        """
        Args:
            imgFile (string): Path of the trainX.npy file
            labelFile (string): Path of the trainY.npy file
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        img_file = np.load(img_file, allow_pickle=True)

        self.img_file = (img_file.reshape(img_file.shape[0], 1, 64, 64))/255
        
        self.label_file = np.load(label_file, allow_pickle=True)
        self.transform = transform

    def __len__(self):
        return len(self.img_file)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = self.img_file[idx]
        label = self.label_file[idx]

        if self.transform:
            sample = self.transform(img)

        #       img  label
        return (img, label)
