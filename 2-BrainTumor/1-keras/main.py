import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from PIL import Image

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

import warnings

warnings.filterwarnings('ignore')

imgs = np.load('trainX.npy', allow_pickle=True)
labels = np.load('trainY.npy', allow_pickle=True)

#print(imgs[20]) # it is 'no' image
#print(labels[20] == [1, 0]) # it give [True True]

# test size is %10
# train size is %90
trainX, valX, trainY, valY = train_test_split(imgs, labels, test_size = 0.1,
                                              random_state = 2)

print('trainX shape', trainX.shape) # imgs
print('valX shape', valX.shape) # imgs
print('trainY shape', trainY.shape) # labels
print('valY shape', valY.shape) # labels

# conv => maxpool => dropout => conv => maxpool => dropout =>
# conv => maxpool => dropout => fully connected
def build_model():
    model = Sequential()

    model.add(Conv2D(filters = 16, kernel_size = (5,5), padding = 'Same',
                     activation = 'relu', input_shape = (64,64,1)))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.25))# %25
    #
    model.add(Conv2D(filters = 32, kernel_size = (3,3), padding = 'Same',
                     activation = 'relu'))
    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
    model.add(Dropout(0.25))# %25
    #
    model.add(Conv2D(filters = 64, kernel_size = (3,3), padding = 'Same',
                     activation = 'relu'))
    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
    model.add(Dropout(0.25))# %25
    
    #fully connected
    model.add(Flatten())
    model.add(Dense(256, activation = 'relu'))
    model.add(Dropout(0.5))# %50
    model.add(Dense(256, activation = 'relu'))
    model.add(Dropout(0.25))# %25
    model.add(Dense(2, activation = 'softmax'))

    return model

# model
model = build_model()

# optimizer
optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)

#compile the model
model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics=['accuracy'])

epochs = 10
batch_size = 3

# data augmentation
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # dimesion reduction
        rotation_range=5,  # randomly rotate images in the range 5 degrees
        zoom_range = 0.1, # Randomly zoom image 10%
        width_shift_range=0.1,  # randomly shift images horizontally 10%
        height_shift_range=0.1,  # randomly shift images vertically 10%
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

datagen.fit(trainX)

history = model.fit_generator(datagen.flow(trainX, trainY, batch_size=batch_size),
                              epochs=epochs, validation_data=(valX, valY),
                              verbose = 2,
                              steps_per_epoch=trainX.shape[0] // batch_size)

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
    
    im = (im.reshape(1, im.shape[0], im.shape[1], 1))/255

    predict = model.predict(im)

    no = int(predict[0][0]*100)
    yes = int(predict[0][1]*100)
    
    if(no >= yes):
        text = "not tumor: {}%\ntumor: {}%".format(str(no), str(yes))
    else:
        text = "tumor: {}%\nnot tumor: {}%".format(str(yes), str(no))
        
    print(text)
    
    #plot the predict and image
    plt.imshow(image)
    plt.title(text, fontsize=15)
    plt.axis("off")

    plt.show()
