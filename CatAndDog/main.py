# 0 - dog
# 1 - cat

import numpy as np
import warnings
from sklearn import linear_model
from PIL import Image
import matplotlib.pyplot as plt

#off the warning errors
warnings.filterwarnings('ignore')

#load train images and labels
trainX = np.load('trainX.npy', allow_pickle=True)
trainY = np.load('trainY.npy', allow_pickle=True)

#load test images and labels
testX = np.load('testX.npy', allow_pickle=True)
testY = np.load('testY.npy', allow_pickle=True)

#find how many images are in the file using shape method
numberOfTrain = trainX.shape[0]
numberOfTest = testX.shape[0]

print(trainX.shape)

# 2d image to 1d image - flatten
xTrainFlatten = trainX.reshape(numberOfTrain, trainX.shape[1] * trainX.shape[2])
xTestFlatten = testX.reshape(numberOfTest, testX.shape[1] * testX.shape[2])

print("train X flatten", xTrainFlatten.shape)
print("test X flatten", xTestFlatten.shape)

#taking transpose for ai
train_x = xTrainFlatten.T
test_x = xTestFlatten.T
train_y = trainY.T
test_y = testY.T

print("x train: ",trainX.shape)
print("x test: ",testX.shape)
print("y train: ",trainY.shape)
print("y test: ",testY.shape, "\n")

# model and train
logreg = linear_model.LogisticRegression(max_iter= 200)
print("accuracy: {} ".format(logreg.fit(train_x.T, train_y.T).score(test_x.T, test_y.T)), "\n")

#for use
while True:
    #take path of image for use in the predict
    imPath = input('path of your image: ')

    size = (64, 64)

    #open image
    image = Image.open(imPath)
    
    try:
        #resize and grayscale image
        im = np.array((image.convert('L')).resize(size))
    except:
        print("there is a problem in the path of the photo")    
        continue

    #image reshape for predict
    reshaped = im.reshape(1, -1)

    predict = logreg.predict(reshaped)

    #predictive interpretation
    if(predict[0] == 0):
        text = "that's dog"
    else:
        text = "that's cat"

    #plot the predict and image
    plt.title(text, fontsize=15)
    plt.imshow(image)

    plt.show()
