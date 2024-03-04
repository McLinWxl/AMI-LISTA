import torch.nn as nn
import torch


class DCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(2, 12, kernel_size=25, padding='same')
        self.conv2 = nn.Conv1d(12, 6, kernel_size=15, padding='same')
        self.conv3 = nn.Conv1d(6, 3, kernel_size=5, padding='same')
        self.conv4 = nn.Conv1d(3, 1, kernel_size=3, padding='same')
        self.bn1 = nn.BatchNorm1d(12)
        self.bn2 = nn.BatchNorm1d(6)
        self.bn3 = nn.BatchNorm1d(3)
        # self.bn4 = nn.BatchNorm1d(1)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x = x.transpose(1, 2)
        x = x.to(torch.float32)
        conv1 = self.relu(self.conv1(x))
        conv1 = self.bn1(conv1)
        conv2 = self.relu(self.conv2(conv1))
        conv2 = self.bn2(conv2)
        conv3 = self.relu(self.conv3(conv2))
        conv3 = self.bn3(conv3)
        conv4 = self.relu(self.conv4(conv3))
        return conv4.transpose(1, 2)

