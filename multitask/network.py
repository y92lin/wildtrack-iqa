import torch
import torch.nn as nn

class NSSADNN(nn.Module):
    def __init__(self):
        super(NSSADNN, self).__init__()
        self.fc1_1 = nn.Linear(42, 256)
        self.fc1_2 = nn.Linear(256, 512)
        self.fc1_3 = nn.Linear(512, 1024)
        self.fc1_4 = nn.Linear(1024, 512)
        self.fc1_5 = nn.Linear(512, 128)
        self.fc1_6 = nn.Linear(128, 64)
        self.fc1_7 = nn.Linear(64, 1)

        self.relu = nn.ReLU()
        self.batchnorm1 = nn.BatchNorm1d(256)
        self.batchnorm2 = nn.BatchNorm1d(512)
        self.batchnorm3 = nn.BatchNorm1d(1024)
        self.batchnorm4 = nn.BatchNorm1d(512)
        self.batchnorm5 = nn.BatchNorm1d(128)
        self.batchnorm6 = nn.BatchNorm1d(64)


    def forward(self, x):

        x = self.fc1_1(x)
        x = self.relu(x)
        x = self.batchnorm1(x)

        x = self.fc1_2(x)
        x = self.relu(x)
        x = self.batchnorm2(x)

        x = self.fc1_3(x)
        x = self.relu(x)
        x = self.batchnorm3(x)

        x = self.fc1_4(x)
        x = self.relu(x)
        x = self.batchnorm4(x)


        x = self.fc1_5(x)
        x = self.relu(x)
        x = self.batchnorm5(x)

        x = self.fc1_6(x)
        x = self.relu(x)
        x = self.batchnorm6(x)

        x = self.fc1_7(x)
        x = torch.sigmoid(x)
        return x