# 0 - dog
# 1 - cat

from keras.models import Sequential
from keras.layers import Dense

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

#off the warning errors
import warnings
warnings.filterwarnings('ignore')

#load train images and labels
trainX = np.load('trainX.npy', allow_pickle=True)
trainY = np.load('trainY.npy', allow_pickle=True)

#find how many images are in the file using shape method
numberOfTrain = trainX.shape[0]

print(trainX.shape)

# 2d image to 1d image - flatten
flatten = trainX.reshape(numberOfTrain, trainX.shape[1] * trainX.shape[2])

print("train X flatten", flatten.shape)

print(len(flatten))
print(len(trainY))

def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu', input_dim = flatten.shape[1]))
    classifier.add(Dense(units = 4, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    return classifier

model = build_classifier()
model.fit(flatten, trainY, epochs=10, verbose=2)
"""
mean = accuracies.mean()
variance = accuracies.std()

print("Accuracy mean: "+ str(mean))
print("Accuracy variance: "+ str(variance))
"""
#for use
while True:
    #take path of image for use in the predict
    imPath = input('path of your image: ')

    size = (64, 64)
    
    try:
        #open image
        image = Image.open(imPath)
    except:
        print("there is a problem in the path of the photo")    
        continue

    #resize and grayscale image
    im = np.array((image.convert('L')).resize(size))
    
    im = im.reshape(1, im.shape[0]*im.shape[1])

    predict = model.predict(im)

    #predictive interpretation
    if(predict[0][0] >= 0.5):
        text = "that's dog"
    else:
        text = "that's cat"

    print(predict)
    
    #plot the predict and image
    plt.imshow(image)
    plt.title(text, fontsize=15)
    plt.axis("off")

    plt.show()

    predict = None
