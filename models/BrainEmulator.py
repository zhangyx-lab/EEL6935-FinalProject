# ---------------------------------------------------------
# Yuxuan Zhang
# Dept. of Electrical and Computer Engineering
# University of Florida
# ---------------------------------------------------------
# U_Net Decoder Module
# ---------------------------------------------------------
import torch
import torch.nn as nn
from dataset import Sample_t
from lib.Context import Context
from .Node import Node
from .config import SCALE, FC_LAYERS
from util.loader import ROI

"""
Hirachy of actual human brain:

Visual Stimuli -> 

"""

class BrainEmulator(nn.Module):
    def __init__(self, ctx: Context, sample: Sample_t):
        super().__init__()
        # Unpack sample
        s, t = sample
        # Expect input to be 1D features
        b, c = s.shape
        # Compute middle layer channels
        ch_m = max(s.shape[1], t.shape[1])
        # Initialize fully connected layers
        fc = []
        for i in range(fc_layers):
            in_features = s.shape[1] if i == 0 else ch_m
            out_features = t.shape[1] if i == fc_layers - 1 else ch_m
            fc.append(nn.LeakyReLU())
            fc.append(nn.Linear(in_features, out_features))
        self.fc = nn.Sequential(*fc)
        # Activation function for -1~1 voxel spike
        self.activation = nn.Tanh()
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(self.parameters())

    def forward(self, x: torch.Tensor, train=None):
        b, h, w = x.shape
        x = x.view((b, -1, h, w))
        # Down sampling layers
        for pool, node in self.layers:
            x = node(x, train=train)
            x = pool(x)
        # Fully connected layers
        x = x.view((b, -1))
        x = self.fc(x)
        # Final activation function
        x = self.activation(x)
        return x
