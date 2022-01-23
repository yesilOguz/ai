# 0 - dogs
# 1 - cats 

import os
from PIL import Image
import numpy as np

# paths
trainCats = "training_set/training_set/cats/"
trainDogs = "training_set/training_set/dogs/"

testCats = "test_set/test_set/cats/"
testDogs = "test_set/test_set/dogs/"

#filenames in the paths
catTraining = [f for f in os.listdir(trainCats) if os.path.isfile(os.path.join(trainCats, f))] # 4001 image of cat
dogTraining = [f for f in os.listdir(trainDogs) if os.path.isfile(os.path.join(trainDogs, f))] # 4006 image of dog

catTest = [f for f in os.listdir(testCats) if os.path.isfile(os.path.join(testCats, f))] # 1012 image of cat
dogTest = [f for f in os.listdir(testDogs) if os.path.isfile(os.path.join(testDogs, f))] # 1013 image of dog


def getArrays(catPath, catNames, dogPath, dogNames):
    numOfCat = len(catNames)
    numOfDog = len(dogNames)

    #take minimum value of the filename array's lengths
    if(numOfCat <= numOfDog):
        loop = numOfCat
    else:
        loop = numOfDog

    print(loop)

    #blank arrays
    ImageArray = []
    catDogArray = []

    size = (64, 64)
    
    for i in range(loop):
        #cat
        if catNames[i].endswith(('.jpg', '.png', 'jpeg')):
            #open image and convert to grayscale then resize it finally convert the numpy array
            ImageArray.append(np.asarray((Image.open(catPath+catNames[i]).convert('L')).resize(size)))
            catDogArray.append(np.asarray(1))

        #dog
        if dogNames[i].endswith(('.jpg', '.png', 'jpeg')):
            #open image and convert to grayscale then resize it finally convert the numpy array
            ImageArray.append(np.asarray((Image.open(dogPath+dogNames[i]).convert('L')).resize(size)))
            catDogArray.append(np.asarray(0))

    print(np.asarray(ImageArray[20]).shape)

    #      Images                labels
    return np.array(ImageArray), np.array(catDogArray)

trainX, trainY = getArrays(trainCats, catTraining, trainDogs, dogTraining)
testX, testY = getArrays(testCats, catTest, testDogs, dogTest)

#save arrays as numpy file
np.save("trainX.npy", trainX)# Images
np.save("trainY.npy", trainY)# Labels

np.save("testX.npy", testX)# Images
np.save("testY.npy", testY)# Labels
