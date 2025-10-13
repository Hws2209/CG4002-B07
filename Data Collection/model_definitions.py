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
    def __init__(self, num_channels, num_classes, sequence_length,
                 conv1_out=6, conv2_out=3, kernel_size_conv=3, pool_size=2,
                 fc1_neurons=64, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(num_channels, conv1_out, kernel_size=kernel_size_conv, padding='same')
        self.bn1 = nn.BatchNorm1d(conv1_out)
        self.conv2 = nn.Conv1d(conv1_out, conv2_out, kernel_size=kernel_size_conv, padding='same')
        self.bn2 = nn.BatchNorm1d(conv2_out)
        self.pool = nn.MaxPool1d(kernel_size=pool_size)
        self.fc1 = nn.Linear(conv2_out * (sequence_length // pool_size), fc1_neurons)
        self.fc2 = nn.Linear(fc1_neurons, num_classes)
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
    def __init__(self, num_channels, num_classes, hidden_size=64, num_layers=1, dropout=0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=num_channels, hidden_size=self.hidden_size,
                            num_layers=self.num_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.hidden_size, num_classes)
    
    def forward(self, x):        
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :] # Take last timestep output
        out = self.dropout(out)
        out = self.fc(out) # Map to class scores
        return out


# MLP Model
class ActionMLP(nn.Module):
    def __init__(self, input_size, num_classes,
                 hidden1=256, hidden2=128, dropout=0.3):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, num_classes)
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
    def __init__(self, input_size, num_classes, hidden_size=64, dropout=0.3):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
