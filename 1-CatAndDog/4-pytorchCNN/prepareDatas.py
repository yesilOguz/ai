# 0 - dogs
# 1 - cats

from PIL import Image
import numpy as np
import os

# paths
trainCats = "dataset/training_set/cats/"
trainDogs = "dataset/training_set/dogs/"

testCats = "dataset/test_set/cats/"
testDogs = "dataset/test_set/dogs/"

#filenames in the paths
catTraining = [f for f in os.listdir(trainCats) if os.path.isfile(os.path.join(trainCats, f))] # 4001 image of cat
dogTraining = [f for f in os.listdir(trainDogs) if os.path.isfile(os.path.join(trainDogs, f))] # 4006 image of dog

catTest = [f for f in os.listdir(testCats) if os.path.isfile(os.path.join(testCats, f))] # 1012 image of cat
dogTest = [f for f in os.listdir(testDogs) if os.path.isfile(os.path.join(testDogs, f))] # 1013 image of dog

def getArrays(catPath, catNames, dogPath, dogNames):
    # empty arrays
    ImgArr = []
    labelArr = []

    size = (64, 64)

    for i in dogNames:
        if i.endswith(('.jpg', '.png', 'jpeg')):
            img = np.asarray((Image.open(dogPath+i).convert('L')).resize(size))
            ImgArr.append(img)

            labelArr.append(np.asarray(0))# dog
        
    for i in catNames:
        if i.endswith(('.jpg', '.png', 'jpeg')):
            img = np.asarray((Image.open(catPath+i).convert('L')).resize(size))
            ImgArr.append(img)

            labelArr.append(np.asarray(1))# cat

    #      Images            Labels
    return np.array(ImgArr), np.array(labelArr)

trainX, trainY = getArrays(trainCats, catTraining, trainDogs, dogTraining)
testX, testY = getArrays(testCats, catTest, testDogs, dogTest)

#save arrays as numpy file
np.save("trainX.npy", trainX)# Images
np.save("trainY.npy", trainY)# Labels

np.save("testX.npy", testX)# Images
np.save("testY.npy", testY)# Labels
