import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from model import CNNModel
from dataset import CatAndDog

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

import warnings

warnings.filterwarnings('ignore')

train = CatAndDog(img_file = 'trainX.npy', label_file = 'trainY.npy')
test = CatAndDog(img_file = 'testX.npy', label_file = 'testY.npy')

# batch_size, epoch and iteration
batch_size = 50
n_iters = 2000

num_epochs = n_iters / (len(train) / batch_size)
num_epochs = int(num_epochs)

trainDatas = DataLoader(train, batch_size=batch_size, shuffle=True)
testDatas = DataLoader(test, batch_size=1, shuffle=True)

# Create model
model = CNNModel()

#Binary Cross Entropy Loss
error = nn.BCELoss()

learning_rate = 0.15
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

count = 0
loss_list = []
iteration_list = []
accuracy_list = []

model = model.float()

for epoch in range(num_epochs):
    print('epoch: {}/{}'.format(epoch+1, num_epochs))
    for i, (train, label) in enumerate(trainDatas):
        # train mode
        model.train()
        
        train = Variable(train)
        label = Variable(label)
        
        # Clear gradients
        optimizer.zero_grad()

        # Forward propagation
        outputs = model(train.float())
        outputs = outputs.reshape(outputs.shape[0])

        # 0 => dog
        # 1 => cat
        
        # Calculate sigmoid and cross entropy loss
        loss = error(outputs, label.float())

        # Calculating gradients
        loss.backward()

        # Update parameters
        optimizer.step()

        count += 1

        if count % 50 == 0:
            # Calculate accuracy
            correct = 0
            total = 0

            #predict test dataset
            for i, (test, testLabel) in enumerate(testDatas):
                # predict mode
                model.eval()
                
                test = Variable(test)
                testLabel = Variable(testLabel)

                # Forward propagation
                outputs = model(test.float())

                # Get test's length
                total = len(testDatas)
                
                if((outputs.data.item() >= 0.5 and testLabel.item() == 1) or
                   (outputs.data.item() < 0.5 and testLabel.item() == 0)):

                    # Total correct predictions
                    correct += 1

            accuracy = 100 * (correct / float(total))

            # store loss and iteration
            loss_list.append(loss.data)
            iteration_list.append(count)
            accuracy_list.append(accuracy)
            
            print('Iteration: {}  Loss: {}  Accuracy: {}%({}/{})'.format(
                count, loss.data, int(accuracy), correct, total))
          
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
    
    if(predict.item() >= 0.5):
        text = "that's cat"
    else:
        text = "that's dog"
        
    print(predict.item())
    
    #plot the predict and image
    plt.imshow(image)
    plt.title(text, fontsize=15)
    plt.axis("off")

    plt.show()
