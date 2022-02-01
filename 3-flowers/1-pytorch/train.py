import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from model import CNNModel
from dataset import flowers

import warnings

def train():
    warnings.filterwarnings('ignore')
    
    # paths
    trainX = 'trainX.npy'
    trainY = 'trainY.npy'

    testX = 'testX.npy'
    testY = 'testY.npy'

    train = flowers(trainX, trainY)
    test = flowers(testX, testY)

    batch_size = 5
    n_iters = 5000
    num_epochs = n_iters / (len(train) / batch_size)
    num_epochs = int(num_epochs)

    trainDatas = DataLoader(train, batch_size=batch_size, shuffle=False)
    testDatas = DataLoader(test, batch_size=1, shuffle=False)

    # Create model
    model = CNNModel()

    # Cross Entropy Loss
    error = nn.BCEWithLogitsLoss()

    learning_rate = 0.1
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    count = 0
    loss_list = []
    iteration_list = []
    accuracy_list = []
    best_correct = 0.0

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

            # forward propagation
            outputs = model(train.float())

            #_, pred = torch.max(outputs, 1)

            loss = error(outputs, label.float())

            loss.backward()
            optimizer.step()

            count += 1

            if count % 50 == 0:
                # Calculate Accuracy
                correct = 0
                total = 0

                # predict test dataset
                for i, (test, testLabel) in enumerate(testDatas):
                    # predict mode
                    model.eval()

                    test = Variable(test)
                    testLabel = Variable(testLabel)

                    # forward propagation
                    outputs = model(test.float())

                    _, pred = torch.max(outputs, 1)
                    _, testLabel = torch.max(testLabel.data, 1)

                    if(testLabel == pred):
                        correct += 1
                    
                    total = len(testDatas)

                accuracy = 100 * (correct / float(total))

                if correct > best_correct:
                    torch.save(model, 'best.model')
                    best_correct = correct

                # store loss and iteration
                loss_list.append(loss.data)
                iteration_list.append(count)
                accuracy_list.append(accuracy)

            if count % 100 == 0:
                print('Iteration: {}  Loss: {}  Accuracy: {}%({}/{})'.format(
                    count, loss.data, int(accuracy), correct, total))

