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
from .Node import Node
from util.run import Context


class Decoder(nn.Module):
    def __init__(self, ctx: Context, sample: Sample_t, fc_layers: int = 1, scale=3):
        super().__init__()
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

    def forward(self, x, train=False):
        x = self.fc(x)
        b, w = x.shape
        x = x.view((b, w, 1, 1))
        for upconv, node in self.layers:
            x = upconv(x)
            x = node(x, train=train)
        b, _, h, w = x.shape
        x = x.view((b, h, w))
        return x
