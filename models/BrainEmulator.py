# ---------------------------------------------------------
# Yuxuan Zhang
# Dept. of Electrical and Computer Engineering
# University of Florida
# ---------------------------------------------------------
# U_Net Decoder Module
# ---------------------------------------------------------
import torch
import torch.nn as nn
from torch import cat
from util.loader import decompose, voxel_count as cnt
from .misc import PowerActivation, HiddenLayers
"""
Hirachy of actual human brain:

                                 | V3  |
Visual Stimuli <-> V1 <-> V2 <-> | V3A | <-> V4
      |                          | V3B |
    LatOcc

"""


class BrainEmulatorForward(nn.Module):
    def __init__(self, in_features: int, bias: bool = True):
        super().__init__()
        # Construct layers
        self.V1  = HiddenLayers(in_features, cnt("V1"), bias=bias)
        self.V2  = HiddenLayers(cnt("V1"), cnt("V2"), bias=bias)
        self.V3  = HiddenLayers(cnt("V2"), cnt("V3"), bias=bias)
        self.V3A = HiddenLayers(cnt("V2"), cnt("V3A"), bias=bias)
        self.V3B = HiddenLayers(cnt("V2"), cnt("V3B"), bias=bias)
        self.V4  = HiddenLayers(cnt("V3", "V3A", "V3B"), cnt("V4"), bias=bias)
        self.LatOcc = HiddenLayers(in_features, cnt("LatOcc"), bias=bias)


    def forward(self, visual: torch.Tensor, train=None):
        V1  = self.V1 (visual)
        V2  = self.V2 (V1)
        V3  = self.V3 (V2)
        V3A = self.V3A(V2)
        V3B = self.V3B(V2)
        V3X = cat([V3, V3A, V3B], dim=1)
        V4  = self.V4 (V3X)
        LatOcc = self.LatOcc(visual)
        # Cat for output
        spike = cat([V1, V2, V3X, V4, LatOcc], dim=1)
        return spike


class BrainEmulatorBackward(nn.Module):
    def __init__(self, out_features: int, bias: bool = True):
        super().__init__()
        # Construct layers
        self.V3 = HiddenLayers(cnt("V3", "V4"), cnt("V3"), bias=bias)
        self.V3A = HiddenLayers(cnt("V3A", "V4"), cnt("V3A"), bias=bias)
        self.V3B = HiddenLayers(cnt("V3B", "V4"), cnt("V3B"), bias=bias)
        self.V2 = HiddenLayers(cnt("V2", "V3", "V3A", "V3B"), cnt("V2"), bias=bias)
        self.V1 = HiddenLayers(cnt("V1", "V2"), cnt("V1"), bias=bias)
        self.comb = HiddenLayers(cnt("V1", "LatOcc"), out_features, bias=bias)

    def forward(self, spike: torch.Tensor):
        V1, V2, V3, V3A, V3B, V4, LatOcc = decompose(spike)
        v3 = cat([
            self.V3(cat([V3, V4], dim=1)),
            self.V3A(cat([V3A, V4], dim=1)),
            self.V3B(cat([V3B, V4], dim=1))
        ], dim=1)
        v2 = self.V2(cat([V2, v3], dim=1))
        v1 = self.V1(cat([V1, v2], dim=1))
        return self.comb(cat([v1, LatOcc], dim=1))
