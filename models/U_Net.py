# ---------------------------------------------------------
# Yuxuan Zhang
# Dept. of Electrical and Computer Engineering
# University of Florida
# ---------------------------------------------------------
# TODO: Add description
# ---------------------------------------------------------
import numpy as np
import torch
import torch.nn as nn
from lib.Module import Module
from dataset import Sample_t
from util.env import DEVICE
import util.args as args
from util.run import Context
# ---------------------------------------------------------
from .Decoder import Decoder
from .Encoder import Encoder
# ---------------------- Train Modes ----------------------
from enum import Enum

# class TRAIN_MODE(Enum):
#     def __str__(self):
#         return self.value
#     def __init__(self, mode=None):
#         if mode is None: return
#         assert mode & self.ALL_MODES, f"Unknown TRAIN_MODE [{mode}]"
#         self.vs = mode & self.VIS_ALONE
#         self.sv = mode & self.SPI_ALONE
#         self.vv = mode & self.VIS_JOINT
#         self.ss = mode & self.SPI_JOINT
# Train either [encoder] OR [decoder] alone
VIS_ALONE = 0b0001  # visual -> [encoder] -> spike
SPI_ALONE = 0b0010  # spike -> [decoder] -> visual
ALL_ALONE = VIS_ALONE | SPI_ALONE
# Train [encoder->decoder] OR [decoder->encoder] jointly
VIS_JOINT = 0b0100  # visual -> [encoder => decoder] -> visual
SPI_JOINT = 0b1000  # spike -> [decoder => encoder] -> spike
ALL_JOINT = VIS_JOINT | SPI_JOINT
# The monster
ALL_MODES = ALL_ALONE | ALL_JOINT
# ---------------------------------------------------------
SCALE = 3
FC_LAYERS = 10
visual_loss = nn.MSELoss().to(DEVICE)
spike_loss = nn.MSELoss().to(DEVICE)


class Model(Module):
    def __init__(self, ctx: Context, device, sample: Sample_t):
        super(Model, self).__init__(device)
        visual, spike = sample
        # =============== Encoder ===============
        ctx.log("model input shape", visual.shape)
        # Encoder
        self.encoder = Encoder(ctx, sample, FC_LAYERS, SCALE)
        out = self.encoder(visual)
        # Report shape
        ctx.log("Encoder output shape", out.shape)
        assert spike.shape == out.shape, f"{spike.shape} != {out.shape}"
        del out
        # =============== Decoder ===============
        ctx.log("Decoder input shape", spike.shape)
        # Decoder
        self.decoder = Decoder(ctx, sample, FC_LAYERS, SCALE)
        out = self.decoder(spike)
        # Report output shape
        ctx.log("Decoder output shape", out.shape)
        assert visual.shape == out.shape, f"{visual.shape} != {out.shape}"
        del out
        # ============== Optimizer ==============
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=args.learning_rate,
            weight_decay=5e-4
        )

    def forward(self, *args):
        raise NotImplementedError

    def iterate_batch(self, ctx: Context, data_point, TRAIN_MODE=False):
        visual, spike, idx = data_point
        # Send both visual and spike date to device
        visual = visual.to(self.device)
        spike = spike.to(self.device)
        # Prediction getters
        LOSS: dict[torch.Tensor] = {}
        PRED = {
            "vs": lambda: (self.encoder(visual), spike_loss, spike),
            "sv": lambda: (self.decoder(spike), visual_loss, visual),
            "vv": lambda: (self.decoder(get("vs")[0]), visual_loss, visual),
            "ss": lambda: (self.encoder(get("sv")[0]), spike_loss, spike),
        }

        def get(k):
            nonlocal PRED
            if callable(PRED[k]):
                PRED[k] = PRED[k]()
            return PRED[k]

        def getLoss(k):
            nonlocal LOSS
            pred, loss, target = get(k)
            print("loss", k, pred.shape, target.shape)
            LOSS[k] = loss(pred, target)
            return pred, LOSS[k]
        # Test mode saves all types of predictions to context
        if not TRAIN_MODE:
            with torch.no_grad():
                # visual -> spike
                vs, vs_loss = getLoss("vs")
                # spike -> visual
                sv, sv_loss = getLoss("sv")
                # visual -> visual
                vv, vv_loss = getLoss("vv")
                # spike -> spike
                ss, ss_loss = getLoss("ss")
            # Save all types of prediction
            ID = f"{idx:04d}"
            np.save(ctx.path / f"{ID}_vs", vs)
            np.save(ctx.path / f"{ID}_sv", sv)
            np.save(ctx.path / f"{ID}_vv", vv)
            np.save(ctx.path / f"{ID}_ss", ss)
            # Collect info to log
            info = [
                ('vs', vs_loss), ('sv', sv_loss),
                ('ss', ss_loss), ('vv', vv_loss)
            ]
            # Log to dedicated report file
            ctx.log(ID, '|', ' | '.join([
                f"{t} {v:.04f}" for t, v in info
            ]), file="results.txt")
        # Train strategy varies according to TRAIN_MODE
        elif TRAIN_MODE & ALL_MODES:
            def step(loss):
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            if TRAIN_MODE & VIS_ALONE:
                vs, vs_loss = getLoss("vs")
                step(vs_loss)
            if TRAIN_MODE & SPI_ALONE:
                sv, sv_loss = getLoss("sv")
                step(sv_loss)
            if TRAIN_MODE & VIS_JOINT:
                vv, vv_loss = getLoss("vv")
                step(vv_loss)
            if TRAIN_MODE & SPI_JOINT:
                ss, ss_loss = getLoss("ss")
                step(ss_loss)
        # return super().iterate_batch(ctx, data_point, TRAIN_MODE)
        else:
            assert False, f"Unknown TRAIN_MODE [{TRAIN_MODE}]"
        # Return runtime info to epoch processor
        for k in LOSS:
            loss: torch.Tensor = LOSS[k]
            LOSS[k] =loss.detach().cpu().numpy()
        return LOSS
