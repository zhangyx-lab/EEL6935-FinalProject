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
from .BrainEmulator import BrainEmulatorBackward
from .config import SCALE, FC_LAYERS


class Decoder(Module):
    def __init__(self, ctx: Context, device, sample: Sample_t, fc_layers=FC_LAYERS, scale=SCALE, train=True):
        super().__init__(device, loss=nn.MSELoss().to(device))
        # Unpack sample
        t, s = sample
        # Compute fc layers' out_channels
        ch_t = int(scale ** log(t.shape[-1], 2))
        # Initialize fully connected layers
        self.fc = BrainEmulatorBackward(ch_t)
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
        b, _, h, w = x.shape
        x = x.view((b, h, w))
        return x
