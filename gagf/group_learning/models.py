import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import MaxNLocator

class TwoLayerNet(nn.Module):
    def __init__(self, group_size, hidden_size=None, nonlinearity='square', init_scale=1.0, output_scale=1.0):
        super(TwoLayerNet, self).__init__()
        self.group_size = group_size
        if hidden_size is None:
            hidden_size = 6 * group_size
        self.hidden_size = hidden_size
        self.nonlinearity = nonlinearity
        self.init_scale = init_scale
        self.output_scale = output_scale
        
        # Initialize parameters 
        self.U = nn.Parameter(self.init_scale * torch.randn(hidden_size, self.group_size) / np.sqrt(2 * self.group_size))  
        self.V = nn.Parameter(self.init_scale * torch.randn(hidden_size, self.group_size) / np.sqrt(2 * self.group_size))  
        self.W = nn.Parameter(self.init_scale * torch.randn(hidden_size, self.group_size) / np.sqrt(self.group_size)) # Second layer weights

    def forward(self, x):
        # First layer (linear and combined)
        x1 = x[:, :self.group_size] @ self.U.T
        x2 = x[:, self.group_size:] @ self.V.T
        x_combined = x1 + x2

        # Apply nonlinearity activation
        if self.nonlinearity == 'relu':
            x_combined = torch.relu(x_combined)
        elif self.nonlinearity == 'square':
            x_combined = x_combined**2
        elif self.nonlinearity == 'linear':
            x_combined = x_combined
        elif self.nonlinearity == 'tanh':
            x_combined = torch.tanh(x_combined)
        elif self.nonlinearity == 'gelu':
            gelu = torch.nn.GELU()
            x_combined = gelu(x_combined)
        else:
            raise ValueError(f"Invalid nonlinearity '{self.nonlinearity}' provided.")

        # Second layer (linear)
        x_out = x_combined @ self.W

        # Feature learning scaling
        x_out *= self.output_scale

        return x_out