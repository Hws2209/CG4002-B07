import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import confusion_matrix
from scipy.stats import skew

# =========================
# Model Definitions
# =========================

# CNN Model
class ActionCNN(nn.Module):
    def __init__(self, numChannels, numClasses, sequenceLength,
                 conv1Out=6, conv2Out=3, kernelSizeConv=3, poolSize=2,
                 fc1Neurons=64, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(numChannels, conv1Out, kernel_size=kernelSizeConv, padding='same')
        self.bn1 = nn.BatchNorm1d(conv1Out)
        self.conv2 = nn.Conv1d(conv1Out, conv2Out, kernel_size=kernelSizeConv, padding='same')
        self.bn2 = nn.BatchNorm1d(conv2Out)
        self.pool = nn.MaxPool1d(kernel_size=poolSize)
        self.fc1 = nn.Linear(conv2Out * (sequenceLength // poolSize), fc1Neurons)
        self.fc2 = nn.Linear(fc1Neurons, numClasses)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# RNN Model
class ActionRNN(nn.Module):
    def __init__(self, numChannels, numClasses, hiddenSize=64, numLayers=1, dropout=0.3):
        super().__init__()
        self.hiddenSize = hiddenSize
        self.numLayers = numLayers
        self.lstm = nn.LSTM(input_size=numChannels, hidden_size=self.hiddenSize,
                            num_layers=self.numLayers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.hiddenSize, numClasses)
    
    def forward(self, x):        
        # Initialize hidden and cell states
        h0 = torch.zeros(self.numLayers, x.size(0), self.hiddenSize).to(x.device)
        c0 = torch.zeros(self.numLayers, x.size(0), self.hiddenSize).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :] # Take last timestep output
        out = self.dropout(out)
        out = self.fc(out) # Map to class scores
        return out


# MLP Model
class ActionMLP(nn.Module):
    def __init__(self, inputSize, numClasses,
                 hidden1=256, hidden2=128, dropout=0.3):
        super().__init__()
        self.fc1 = nn.Linear(inputSize, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, numClasses)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# MLP Model with summarised data
class SimplifiedMLP(nn.Module):
    def __init__(self, inputSize, numClasses, hiddenSize=64, dropout=0.3):
        super().__init__()
        self.fc1 = nn.Linear(inputSize, hiddenSize)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hiddenSize, numClasses)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
