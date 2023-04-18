# ---------------------------------------------------------
# Yuxuan Zhang
# Dept. of Electrical and Computer Engineering
# University of Florida
# ---------------------------------------------------------
# U_Net Decoder Module
# ---------------------------------------------------------
from math import log
import torch
import torch.nn as nn
from dataset import Sample_t
from lib.Module import Module
from lib.Context import Context
from util.optimizer import optimizer
from .Node import Node
from .config import SCALE, FC_LAYERS


class Decoder(Module):
    def __init__(self, ctx: Context, device, sample: Sample_t, fc_layers=FC_LAYERS, scale=SCALE, train=True):
        super().__init__(device, loss=nn.MSELoss().to(device))
        # Unpack sample
        t, s = sample
        # Compute fc layers' out_channels
        ch_t = int(scale ** log(t.shape[-1], 2))
        # Compute middle layer channels
        ch_m = max(s.shape[1], ch_t)
        # Initialize fully connected layers
        fc = []
        for i in range(fc_layers):
            in_features = s.shape[1] if i == 0 else ch_m
            out_features = ch_t if i == fc_layers - 1 else ch_m
            fc.append(nn.LeakyReLU())
            fc.append(nn.Linear(in_features, out_features))
        self.fc = nn.Sequential(*fc)
        s = self.fc(s)
        ctx.log("FC output shape", s.shape)
        # Transform view
        b, w = s.shape
        s = s.view((b, w, 1, 1))
        c = w
        # Decompose packed tensors
        layers = []
        # Generate layers
        while c >= scale:
            # Up sample
            upconv = nn.ConvTranspose2d(c, c, 2, 2)
            s = upconv(s)
            # Decoder node
            node = Node(c, int(c / scale))
            s: torch.Tensor = node(s)
            _, c, _, _ = s.shape
            ctx.log("Decoder node shape", s.shape)
            layers.append(nn.ModuleList([upconv, node]))
        self.layers = nn.ModuleList(layers)
        # Activation function for 0~1 grayscale image
        self.activation = nn.Sigmoid()
        # Initialize optimizer
        if train:
            self.optimizer = optimizer(self)

    def iterate_batch(self, ctx: Context, visual, spike, *args, train=False):
        return super().iterate_batch(ctx, spike, visual, *args, train=train)

    def forward(self, x, train=None):
        x = self.fc(x)
        b, w = x.shape
        x = x.view((b, w, 1, 1))
        for upconv, node in self.layers:
            x = upconv(x)
            x = node(x, train=train)
        x = self.activation(x)
        b, _, h, w = x.shape
        x = x.view((b, h, w))
        return x
