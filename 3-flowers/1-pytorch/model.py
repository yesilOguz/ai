import torch
import torch.nn as nn

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=1)

        self.norm1 = nn.BatchNorm2d(64)

        self.maxpool1 = nn.MaxPool2d(3)

        self.relu1 = nn.LeakyReLU()

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=1, padding=1)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=5, stride=1, padding=1)

        self.norm2 = nn.BatchNorm2d(256)
        
        self.maxpool2 = nn.MaxPool2d(3)
        
        self.relu2 = nn.LeakyReLU()

        self.fc1 = nn.Linear(43264, 312)
        self.fc2 = nn.Linear(312, 300)
        self.fc3 = nn.Linear(300, 150)
        self.fc4 = nn.Linear(150, 75)
        self.fc5 = nn.Linear(75, 32)
        self.fc6 = nn.Linear(32, 5)

        self.softmax = nn.Softmax()

    def forward(self, x):
        out = self.conv1(x)        
        out = self.conv2(out)        
        out = self.conv3(out)

        out = self.norm1(out)
        
        out = self.maxpool1(out)

        out = self.relu1(out)

        out = self.conv4(out)        
        out = self.conv5(out)        
        out = self.conv6(out)

        out = self.norm2(out)

        out = self.relu2(out)
        
        # flatten
        out = out.view(out.size(0), -1)
        
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.fc4(out)
        out = self.fc5(out)
        out = self.softmax(self.fc6(out))

        return out
