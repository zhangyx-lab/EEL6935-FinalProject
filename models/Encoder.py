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
from lib.Module import Module
from lib.Context import Context
from util.augment import affine
from util.optimizer import optimizer
from .Node import Node
from .config import SCALE, FC_LAYERS


class Encoder(Module):
    def __init__(self, ctx: Context, device, sample: Sample_t, fc_layers=FC_LAYERS, scale=SCALE, train=True):
        super().__init__(device, loss=nn.HuberLoss().to(device))
        # Unpack sample
        s, t = sample
        # Add 4th dimension to the sample
        # Get channels of input sample
        b, h, w = s.shape
        assert h == w, (h, w)
        s = s.view((-1, 1, h, w))
        c = 1
        # Initialize downscale conv nodes
        layers = []
        # Down-scaler
        pool = nn.MaxPool2d((2, 2))
        # Generate node list according to input sample and channels
        while h >= 2:
            # Get dimensions out of the current sample
            # Create new node layer using the sample
            node = Node(c, scale * c)
            # Iterate the sample
            s = node(s)
            # Down-sample the image
            s: torch.Tensor = pool(s)
            # Record the shape
            _, c, h, w = s.shape
            ctx.log("Encoder node shape", s.shape)
            # Append layer to node list
            layers.append(nn.ModuleList([pool, node]))
        # Instantiate node list
        self.layers = nn.ModuleList(layers)
        # Flatten sample
        s = s.view((b, -1))
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
        # self.activation = nn.Tanh()
        # Initialize optimizer
        if train:
            self.optimizer = optimizer(self)

    def iterate_batch(self, ctx: Context, *data_point, train=None):
        if train and "AFFINE" in train:
            visual, spike = list(data_point)[:2]
            visual = affine(visual)
            return super().iterate_batch(ctx, visual, spike, train=train)
        else:
            return super().iterate_batch(ctx, *data_point, train=train)

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
        # x = self.activation(x)
        return x
