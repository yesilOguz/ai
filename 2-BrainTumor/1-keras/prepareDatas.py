# [1, 0] - no
# [0, 1] - yes

from PIL import Image
import numpy as np
import os

# paths
pathOfNo = 'dataset/no/'
pathOfYes = 'dataset/yes/'

# filenames in the path
noNames = [f for f in os.listdir(pathOfNo) if os.path.isfile(os.path.join(pathOfNo, f))]
yesNames = [f for f in os.listdir(pathOfYes) if os.path.isfile(os.path.join(pathOfYes, f))]

def getArrays(noPath, noNames, yesPath, yesNames):
    # empty arrays
    ImgArr = []
    labelArr = []

    size = (64, 64)

    for i in noNames:
        img = np.asarray((Image.open(noPath+i).convert('L')).resize(size))
        
        # compress the image between 0 and 1. add channel
        # cnns want 3D images
        ImgArr.append((img.reshape(img.shape[0], img.shape[1], 1))/255)

        labelArr.append(np.asarray([1, 0]))# no
        
    for i in yesNames:
        img = np.asarray((Image.open(yesPath+i).convert('L')).resize(size))
        
        # compress the image between 0 and 1. add channel
        # cnns want 3D images
        ImgArr.append((img.reshape(img.shape[0], img.shape[1], 1))/255)

        labelArr.append(np.asarray([0, 1]))# yes

    print(np.asarray(ImgArr[20]).shape)

    #      Images            Labels
    return np.array(ImgArr), np.array(labelArr)

x, y = getArrays(pathOfNo, noNames, pathOfYes, yesNames)

# save arrays as numpy file
np.save('trainX.npy', x) # Images
np.save('trainY.npy', y) # Labels
