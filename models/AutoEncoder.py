# ---------------------------------------------------------
# Yuxuan Zhang
# Dept. of Electrical and Computer Engineering
# University of Florida
# ---------------------------------------------------------
# TODO: Add description
# ---------------------------------------------------------
import torch
import torch.nn as nn
from dataset import Sample_t
from util.env import Path
from lib.Context import Context
from lib.Module import Module
# ---------------------------------------------------------
from .Decoder import Decoder
from .Encoder import Encoder
from .config import SCALE, FC_LAYERS


class Model(Module):
    def __init__(self, ctx: Context, device, sample: Sample_t):
        super(Model, self).__init__(device, loss=nn.MSELoss().to(device))
        visual, spike = sample
        # =============== Encoder ===============
        self.encoder = Encoder(ctx, device, sample, FC_LAYERS, SCALE)
        out = self.encoder(visual)
        # Report shape
        assert spike.shape == out.shape, f"{spike.shape} != {out.shape}"
        del out
        # =============== Decoder ===============
        self.decoder = Decoder(ctx, device, sample, FC_LAYERS, SCALE)
        out = self.decoder(spike)
        # Report output shape
        ctx.log("Decoder output shape", out.shape)
        assert visual.shape == out.shape, f"{visual.shape} != {out.shape}"
        del out
        # ============== Optimizer ==============
        self.optimizer = torch.optim.Adam(self.parameters())

    def save(self, ctx: Context, path: Path):
        self.encoder.save(ctx, path)
        self.decoder.save(ctx, path)

    def load(self, ctx: Context, *path_list: Path):
        # Check length of path list
        if len(path_list) == 1:
            encoder_load_path = path_list[0]
            decoder_load_path = path_list[0]
        else:
            assert len(path_list) == 2, path_list
            encoder_load_path = path_list[0]
            decoder_load_path = path_list[1]

        self.encoder.load(ctx, encoder_load_path)
        self.decoder.load(ctx, decoder_load_path)

    def iterate_batch(self, ctx: Context, *data_point, train=None):
        if train:
            return super().iterate_batch(ctx, *data_point, train=train)
        else:
            visual, spike = list(data_point)[:2]
            report = {}
            with torch.no_grad():
                encoder_pred = self.encoder(visual)
                loss = self.encoder.loss(visual, spike)
                # report[]
                VisualAE_pred = self.decoder(encoder_pred)
                dec_pred = self.decoder(spike)


class VisualAE(Model):
    def iterate_batch(self, ctx: Context, *data_point, train=None):
        if train:
            visual, spike = list(data_point)[:2]
            return super().iterate_batch(ctx, visual, visual, train=train)
        else:
            return super().iterate_batch(ctx, *data_point)

    def forward(self, visual: torch.Tensor, train=None):
        spike = self.encoder(visual, train=train)
        visual = self.decoder(spike, train=train)
        return visual


class SpikeAE(Model):
    def iterate_batch(self, ctx: Context, *data_point, train=None):
        if train:
            visual, spike = list(data_point)[:2]
            return super().iterate_batch(ctx, spike, spike, train=train)
        else:
            return super().iterate_batch(ctx, *data_point)

    def forward(self, spike: torch.Tensor, train=None):
        visual = self.decoder(spike, train=train)
        spike = self.encoder(visual, train=train)
        return spike