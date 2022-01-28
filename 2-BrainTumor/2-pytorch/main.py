import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import imgaug.augmenters as iaa

from model import CNNModel
from dataset import Tumor

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

import warnings

warnings.filterwarnings('ignore')

aug = iaa.Sequential([
    iaa.Superpixels(p_replace=0.5, n_segments=64),
    iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}),
    ])

train = Tumor(isTrain = True, imgFile = 'trainX.npy', labelFile = 'trainY.npy', transform = aug)
test = Tumor(isTrain = False, imgFile = 'trainX.npy', labelFile = 'trainY.npy')

# batch_size, epoch and iteration
batch_size = 5
n_iters = 1500
num_epochs = n_iters / (len(train) / batch_size)
num_epochs = int(num_epochs)

trainDatas = DataLoader(train, batch_size=batch_size, shuffle=True)
testDatas = DataLoader(test, batch_size=1, shuffle=True)

# Create model
model = CNNModel()

# Cross Entropy Loss
error = nn.CrossEntropyLoss()

learning_rate = 0.1
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

count = 0
loss_list = []
iteration_list = []
accuracy_list = []

model = model.float()

for epoch in range(num_epochs):
    print('epoch: {}/{}'.format(epoch+1, num_epochs))
    for (train, label) in trainDatas:
        # train mode
        model.train()
        
        train = Variable(train)
        label = Variable(label)
        
        # Clear gradients
        optimizer.zero_grad()

        # Forward propagation
        outputs = model(train.float())

        # [1, 0] => no
        # [0, 1] => yes
        
        # Calculate softmax and cross entropy loss
        loss = error(outputs, label.long())

        # Calculating gradients
        loss.backward()

        # Update parameters
        optimizer.step()

        count += 1

        if count % 10 == 0:
            # Calculate accuracy
            correct = 0
            total = 0

            #predict test dataset
            for (test, testLabel) in testDatas:
                # predict mode
                model.eval()
                
                test = Variable(test)
                testLabel = Variable(testLabel)

                # Forward propagation
                outputs = model(test.float())

                # Get test's length
                total = len(testDatas)
                
                if((outputs.data[0][1].item() >= outputs.data[0][0].item() and testLabel.item() == 1) or
                   (outputs.data[0][1].item() < outputs.data[0][0].item() and testLabel.item() == 0)):

                    # Total correct predictions
                    correct += 1

            accuracy = 100 * (correct / float(total))

            # store loss and iteration
            loss_list.append(loss.data)
            iteration_list.append(count)
            accuracy_list.append(accuracy)

            print('Iteration: {}  Loss: {}  Accuracy: {} %'.format(
                count, loss.data, int(accuracy)))
            
while True:
    model.eval()
    
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
    
    im = torch.Tensor((im.reshape(1, 1, im.shape[0], im.shape[1]))/255)
    im = Variable(im)

    predict = model(im.float())
    
    no = int(predict[0][0].item()*100)
    yes = int(predict[0][1].item()*100)
    
    if(no > yes):
        text = "not tumor: {}%\ntumor: {}%".format(str(no), str(yes))
    else:
        text = "tumor: {}%\nnot tumor: {}%".format(str(yes), str(no))
        
    print(text)
    
    #plot the predict and image
    plt.imshow(image)
    plt.title(text, fontsize=15)
    plt.axis("off")

    plt.show()
