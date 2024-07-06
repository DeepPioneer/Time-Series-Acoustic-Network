import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt

def num_flat_features(x):
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
        num_features *= s
    return num_features

class WaveMsNet(nn.Module):
    def __init__(self):
        super(WaveMsNet, self).__init__()
        self.conv1_1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=11, stride=1, padding=5)
        self.conv1_2 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=51, stride=5, padding=25)
        self.conv1_3 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=101, stride=10, padding=50)

        self.bn1_1 = nn.BatchNorm1d(32)
        self.bn1_2 = nn.BatchNorm1d(32)
        self.bn1_3 = nn.BatchNorm1d(32)

        self.conv2_1 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=11, stride=1, padding=5)
        self.conv2_2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=11, stride=1, padding=5)
        self.conv2_3 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=11, stride=1, padding=5)

        self.bn2_1 = nn.BatchNorm1d(32)
        self.bn2_2 = nn.BatchNorm1d(32)
        self.bn2_3 = nn.BatchNorm1d(32)

        self.pool2_1 = nn.MaxPool1d(kernel_size=150, stride=150)
        self.pool2_2 = nn.MaxPool1d(kernel_size=30, stride=30)
        self.pool2_3 = nn.MaxPool1d(kernel_size=15, stride=15)

        self.conv3 = nn.Conv1d(in_channels=96, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(2)

        self.conv4 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm1d(256)
        self.pool4 = nn.MaxPool1d(2)

        self.fc1 = nn.Linear(1536, 512)
        self.fc2 = nn.Linear(512, 5)

        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

        self.gradients = None  # Placeholder for gradients
        self.feature_maps = None  # Placeholder for feature maps

    def forward(self, x):
        x1 = self.relu(self.bn1_1(self.conv1_1(x)))
        x2 = self.relu(self.bn1_2(self.conv1_2(x)))
        x3 = self.relu(self.bn1_3(self.conv1_3(x)))

        x1 = self.relu(self.bn2_1(self.conv2_1(x1)))
        x2 = self.relu(self.bn2_2(self.conv2_2(x2)))
        x3 = self.relu(self.bn2_3(self.conv2_3(x3)))

        x1 = self.pool2_1(x1)
        x2 = self.pool2_2(x2)
        x3 = self.pool2_3(x3)

        h = torch.cat((x1, x2, x3), dim=1)

        h = self.conv3(h)
        h = self.bn3(h)
        h = self.relu(h)
        h = self.pool3(h)

        h = self.conv4(h)
        h = self.bn4(h)
        h = self.relu(h)
        h = self.pool4(h)

        self.feature_maps = h  # Save the feature maps

        h.register_hook(self.save_gradient)  # Register hook to save gradients

        h = h.view(-1, num_flat_features(h))
        h = F.relu(self.fc1(h))
        h = self.dropout(h)
        h = self.fc2(h)
        return h

    def save_gradient(self, grad):
        self.gradients = grad


