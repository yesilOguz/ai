import torch
import torch.nn as nn

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5)

        self.norm1 = nn.InstanceNorm2d(32)

        self.maxpool1 = nn.MaxPool2d(3)

        self.relu1 = nn.LeakyReLU()

        self.conv4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5)
        self.conv5 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5)
        self.conv6 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5)

        self.norm2 = nn.InstanceNorm2d(128)

        self.maxpool2 = nn.MaxPool2d(3)

        self.relu2 = nn.LeakyReLU()

        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 60)
        self.fc3 = nn.Linear(60, 30)
        self.fc4 = nn.Linear(30, 20)
        self.fc5 = nn.Linear(20, 10)
        self.fc6 = nn.Linear(10, 5)
        self.fc7 = nn.Linear(5, 1)

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

        out = self.maxpool2(out)

        out = self.relu2(out)

        # flatten
        out = out.view(out.size(0), -1)
        
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.fc4(out)
        out = self.fc5(out)
        out = self.fc6(out)
        out = self.fc7(out)
        
        out = torch.sigmoid(out)

        return out
