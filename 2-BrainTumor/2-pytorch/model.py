import torch
import torch.nn as nn

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        # Conv 1
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5,
                              stride=1, padding=0)
        self.relu1 = nn.ReLU()

        # Dropout2d 1
        self.d1 = nn.Dropout2d(p=0.2)

        # Max pool 1
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        # Batch Normalization 1
        self.bn1 = nn.BatchNorm2d(16)

        # Conv 2
        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5,
                              stride=1, padding=0)
        self.relu2 = nn.ReLU()

        # Dropout2d 2
        self.d2 = nn.Dropout2d(p=0.2)

        # Max pool 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        # Batch Normalization 2
        self.bn2 = nn.BatchNorm2d(32)

        # Fully connected
        self.fc1 = nn.Linear(5408, 1500)

        # Dropout 1
        self.dp1 = nn.Dropout(p=0.2)

        # Fully connected 2
        self.fc2 = nn.Linear(1500, 250)

        # Dropout 2
        self.dp2 = nn.Dropout(p=0.2)
        
        # Fully connected 3
        self.fc3 = nn.Linear(250, 80)
        
        # Dropout 3
        self.dp3 = nn.Dropout(p=0.2)

        # Fully connected 4
        self.fc4 = nn.Linear(80, 40)

        # Fully connected 5
        self.fc5 = nn.Linear(40, 2)

    def forward(self, x):
        # Conv 1
        out = self.cnn1(x)
        out = self.relu1(out)

        # Dropout 2d 1
        out = self.d1(out)

        # max pool 1
        out = self.maxpool1(out)

        # batch normalization
        out = self.bn1(out)

        # Conv 2
        out = self.cnn2(out)
        out = self.relu2(out)

        # Dropout 2d 2
        out = self.d2(out)

        # max pool 2
        out = self.maxpool2(out)

        # batch normalization
        out = self.bn2(out)
        
        # flatten
        out = out.view(out.size(0), -1)

        # Linear func
        out = self.fc1(out)
        # Dropout 1
        out = self.dp1(out)
        out = self.fc2(out)
        # Dropout 2
        out = self.dp2(out)
        out = self.fc3(out)
        # Dropout 3
        out = self.dp3(out)
        out = self.fc4(out)
        out = torch.sigmoid(self.fc5(out))

        return out
