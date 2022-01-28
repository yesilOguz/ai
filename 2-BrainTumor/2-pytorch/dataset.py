# 0 for no
# 1 for yes

import torch
from torch.utils.data import Dataset

from sklearn.model_selection import train_test_split

import numpy as np

class Tumor(Dataset):
    def __init__(self, isTrain, imgFile, labelFile, transform=None):
        """
        Args:
            imgFile (string): Path of the trainX.npy file
            labelFile (string): Path of the trainY.npy file
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.isTrain = isTrain
        
        imgs = np.load(imgFile, allow_pickle=True)
        imgs = imgs.reshape(imgs.shape[0], 1, imgs.shape[1], imgs.shape[2])
        
        labels = np.load(labelFile, allow_pickle=True)
        
        #trainX is images, testX is image, trainY is label, testY is label
        self.trainX, self.testX, self.trainY, self.testY = train_test_split(imgs, labels, test_size = 0.1,
                                              random_state = 2)
        
        self.transform = transform

    def __len__(self):
        if self.isTrain:
            return len(self.trainX)
        else:
            return len(self.testX)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.isTrain:
            trainImg = self.trainX[idx]
            trainLabel = self.trainY[idx]

            if trainLabel[0] == 0:
                trainLabel = 0
            else:
                trainLabel = 1

            if self.transform:
                trainImg = self.transform(image=trainImg)

            #      img       label
            return (trainImg, trainLabel)

        else:
            testImg = self.testX[idx]
            testLabel = self.testY[idx]

            if testLabel[0] == 0:
                testLabel = 0
            else:
                testLabel = 1
            
            #      img      label
            return (testImg, testLabel)
