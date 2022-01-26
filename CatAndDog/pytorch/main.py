import torch
import torch.nn as nn
from torch.autograd import Variable

from model import ANNModel
from dataset import CatAndDog

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# instantiate ANN
input_dim = 64*64
hidden_dim = 200 #hidden layer dim is one of the hyper parameter and it should be chosen and tuned. For now I only say 200 there is no reason.
output_dim = 1

# Create ANN
model = ANNModel(input_dim, hidden_dim, output_dim)

# Binary Cross Entropy Loss 
error = nn.BCELoss()

# Adam Optimizer
learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# ANN model training
count = 0
loss_list = []
iteration_list = []
accuracy_list = []

catDogTrain = CatAndDog(img_file='trainX.npy', label_file='trainY.npy')
catDogTest = CatAndDog(img_file='testX.npy', label_file='testY.npy')

# ANN model training
count = 0
loss_list = []
iteration_list = []
accuracy_list = []
for i in range(len(catDogTrain)):
    sample = catDogTrain[i]
    train = Variable(torch.Tensor(sample['img']))
    label = Variable(torch.Tensor(sample['label']))

    if label.nelement() == 0:
        label = torch.zeros(1,1).reshape(1)

    # Clear gradients
    optimizer.zero_grad()

    # Forward propagation
    outputs = model(train)
    
    # Calculate softmax and ross entropy loss
    loss = error(outputs.data, label)
    loss.requires_grad = True

    # Calculating gradients
    loss.backward()

    # Update parameters
    optimizer.step()

    count += 1

    if count % 50 == 0:
        # Calculate accuracy
        correct = 0
        total = 0
        accuracy = 0
        
        #predict test dataset
        for j in range(len(catDogTest)):
            testSample = catDogTest[j]
            test = Variable(torch.Tensor(testSample['img']))
            testLabel = Variable(torch.Tensor(testSample['label']))

            # Forward propagation
            outputs = model(test)

            # Get data's length
            total = len(catDogTest)

            if((testSample['label'] == 1 and outputs.item() >= 0.5) or
               (testSample['label'] == 0 and outputs.item() < 0.5)):
            
                # Total correct predictions
                correct += 1

        accuracy = 100 * correct / float(total)

        #store loss and iteration
        loss_list.append(loss.data)
        iteration_list.append(count)
        accuracy_list.append(accuracy)

    if count % 500 == 0:
        # print loss
        print('Iteration: {}  Loss: {}  Accuracy: {} %'.format(
            count, loss.data, accuracy))

while True:
    #take path of image for use in the predict
    imPath = input('path of your image: ')

    size = (64, 64)

    try:
        #open image
        image = Image.open(imPath)
    except Exception as e:
        print("there is a problem in the path of the photo: err: " , e)    
        continue

    #resize and grayscale image
    im = np.array((image.convert('L')).resize(size))
    
    flat = im.reshape(im.shape[0]*im.shape[1])
    
    im = Variable(torch.Tensor(flat))

    predict = model(im)

    #predictive interpretation
    if(predict.item() >= 0.5):
        text = "that's dog"
    else:
        text = "that's cat"

    print(predict.item())
    
    #plot the predict and image
    plt.imshow(image)
    plt.title(text, fontsize=15)
    plt.axis("off")
    
    plt.show()
        
