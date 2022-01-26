import torch
from torch.utils.data import Dataset

import numpy as np

class CatAndDog(Dataset):
    def __init__(self, img_file, label_file, transform=None):
        """
        Args:
            npy_file (string): Path of the trainX.npy file
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        
        self.img_file = np.load(img_file, allow_pickle=True)        
        self.label_file = np.load(label_file, allow_pickle=True)
        self.transform = transform

    def __len__(self):
        return self.img_file.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_file = self.img_file
        flat = img_file.reshape(img_file.shape[0],
                                img_file.shape[1]*img_file.shape[2])
        
        img = flat[idx]
        label = self.label_file[idx]

        sample = {'img': img, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample
        
        
        
