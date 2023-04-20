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
from util.optimizer import optimizer
from util.env import Path
from lib.Context import Context
from lib.Module import Module
# ---------------------------------------------------------
from .Decoder import Decoder
from .Encoder import Encoder
from .config import SCALE, FC_LAYERS
# ---------------------------------------------------------
import cv2
from .misc import test_set, train_set


class Model(Module):
    def __init__(self, ctx: Context, device, sample: Sample_t):
        super(Model, self).__init__(device, loss=nn.MSELoss().to(device))
        visual, spike = sample
        # =============== Encoder ===============
        self.encoder = Encoder(ctx, device, sample, FC_LAYERS, SCALE, train=False)
        out = self.encoder(visual)
        # Report shape
        assert spike.shape == out.shape, f"{spike.shape} != {out.shape}"
        del out
        # =============== Decoder ===============
        self.decoder = Decoder(ctx, device, sample, FC_LAYERS, SCALE, train=False)
        out = self.decoder(spike)
        # Report output shape
        ctx.log("Decoder output shape", out.shape)
        assert visual.shape == out.shape, f"{visual.shape} != {out.shape}"
        del out
        # ============== Optimizer ==============
        self.optimizer = optimizer(self)

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

    def iterate_epoch(self, epoch, loader, ctx: Context, train=None):
        result = super().iterate_epoch(epoch, loader, ctx, train)
        with torch.no_grad():
            for ds, set in ((test_set, 'test'), (train_set, 'train')):
                sample = ds.sample(5)
                encoder_preview = self.encoder.preview(sample)
                decoder_preview = self.decoder.preview(sample)
                cv2.imwrite(str(ctx.path / f"encoder-{set}.png"), encoder_preview)
                cv2.imwrite(str(ctx.path / f"decoder-{set}.png"), decoder_preview)
        return result

    def iterate_batch(self, ctx: Context, *data_point, train=None):
        if train:
            return super().iterate_batch(ctx, *data_point, train=train)
        else:
            visual, spike = list(data_point)[:2]
            visual = visual.to(self.device)
            spike = spike.to(self.device)
            report = {}
            with torch.no_grad():
                # Encoder driven
                spike_pred = self.encoder(visual)
                loss = self.encoder.loss(spike_pred, spike)
                report["encoder loss"] = loss.detach().cpu().numpy()
                pred = spike_pred.detach().cpu().numpy()
                ctx.push('Encoder.prediction', pred)

                visual_pred = self.decoder(spike_pred)
                loss = self.decoder.loss(visual_pred, visual)
                report["enc-dec loss"] = loss.detach().cpu().numpy()
                pred = visual_pred.detach().cpu().numpy()
                ctx.push('VisualAE.prediction', pred)

                del spike_pred, visual_pred, loss

                # Decoder driven
                visual_pred = self.decoder(spike)
                loss = self.decoder.loss(visual_pred, visual)
                report["decoder loss"] = loss.detach().cpu().numpy()
                pred = visual_pred.detach().cpu().numpy()
                ctx.push('Decoder.prediction', pred)

                spike_pred = self.encoder(visual_pred)
                loss = self.encoder.loss(spike_pred, spike)
                report["dec-enc loss"] = loss.detach().cpu().numpy()
                pred = spike_pred.detach().cpu().numpy()
                ctx.push('SpikeAE.prediction', pred)

                del spike_pred, visual_pred, loss

            return report


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
    def lossFunction(self, pred, truth):
        return super().lossFunction(pred, truth)

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