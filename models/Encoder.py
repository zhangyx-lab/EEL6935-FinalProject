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
from .BrainEmulator import BrainEmulatorForward
from .config import SCALE, FC_LAYERS
# Used for saving preview
import cv2
from util.visualize import spike as visualize
from .misc import test_set, train_set

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
            # Create new node layer using the sample
            node = Node(c, scale * c)
            # Iterate the sample
            s = node(s)
            # Down-sample the image
            s: torch.Tensor = pool(s)
            # Get dimensions out of the current sample
            _, c, h, w = s.shape
            # Record the shape
            ctx.log("Encoder node shape", s.shape)
            # Append layer to node list
            layers.append(nn.ModuleList([node, pool]))
        # Instantiate node list
        self.layers = nn.ModuleList(layers)
        # Flatten sample
        s = s.view((b, -1))
        # Initialize fully connected layers
        self.fc = BrainEmulatorForward(in_features=s.shape[1])
        # self.fc = HiddenLayers(s.shape[1], t.shape[1], delta=0.5, middle=1.2)
        # Get final sample output
        s = self.fc(s)
        self.activation = nn.Sigmoid()
        print("Encoder output shape", s.shape)
        # Initialize optimizer
        if train:
            self.optimizer = optimizer(self)

    def preview(self, sample):
        visual, spike = sample
        pred = self(visual.to(self.device)).detach().cpu().numpy()
        return visualize(spike, pred)

    def iterate_epoch(self, epoch, loader, ctx: Context, train=None):
        result = super().iterate_epoch(epoch, loader, ctx, train)
        with torch.no_grad():
            for ds, name in ((test_set, 'test'), (train_set, 'train')):
                preview = self.preview(ds.sample(5))
                savepath = str(ctx.path / f"preview-{name}.png")
                cv2.imwrite(savepath, preview)
        return result

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
        for node, pool in self.layers:
            x = node(x, train=train)
            x = pool(x)
        # Fully connected layers
        x = self.fc(x.view((b, -1)))
        return self.activation(x)
