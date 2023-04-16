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
from lib.Context import Context
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
VIS_ALONE = 0b0001  # visual -> [encoder] -> spike
SPI_ALONE = 0b0010  # spike -> [decoder] -> visual
VIS_JOINT = 0b0100  # visual -> [encoder => decoder] -> visual
SPI_JOINT = 0b1000  # spike -> [decoder => encoder] -> spike
ALL_ALONE = VIS_ALONE | SPI_ALONE
ALL_JOINT = VIS_JOINT | SPI_JOINT
ALL_MODES = ALL_ALONE | ALL_JOINT
# ---------------------------------------------------------
SCALE = 3
FC_LAYERS = 4
visual_loss = nn.MSELoss().to(DEVICE)
spike_loss = nn.MSELoss().to(DEVICE)
optim_visual = None
optim_spike = None


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
        global optim_visual, optim_spike
        optim_visual, optim_spike = [
            torch.optim.Adam(
                self.parameters(),
                lr=args.learning_rate,
                weight_decay=5e-4
            ) for _ in range(2)
        ]
        # Not used
        self.optimizer = torch.optim.Adam(self.parameters())

    def forward(self, *args):
        raise NotImplementedError

    def iterate_batch(self, ctx: Context, data_point, TRAIN_MODE=False):
        visual, spike, idx = data_point
        # Send both visual and spike date to device
        visual = visual.to(self.device)
        spike = spike.to(self.device)
        # Dictionary that saves all detached losses
        LOSS: dict[torch.Tensor] = {}
        PRED = {
            "V-S": lambda: (self.encoder(visual), spike_loss, spike),
            "S-V": lambda: (self.decoder(spike), visual_loss, visual),
            "V-V": lambda: (self.decoder(get("V-S")[0]), visual_loss, visual),
            "S-S": lambda: (self.encoder(get("S-V")[0]), spike_loss, spike),
        }

        def get(k):
            nonlocal PRED
            if callable(PRED[k]):
                PRED[k] = PRED[k]()
            return PRED[k]

        def getLoss(k, detach=False) -> tuple[torch.Tensor, torch.Tensor]:
            nonlocal LOSS
            pred, lossFn, target = get(k)
            # print("loss", k, pred.shape, target.shape)
            loss: torch.Tensor = lossFn(pred, target)
            LOSS[k] = loss.detach().cpu().numpy()
            if detach:
                del loss
                return pred, LOSS[k]
            else:
                return pred, loss
        # Test mode saves all types of predictions to context
        if not TRAIN_MODE:
            ID = f"{int(idx):04d}"

            def save(suffix: str, tensor: torch.Tensor):
                np.save(ctx.path / f"{ID}_{suffix}",
                        tensor.detach().cpu().numpy())
            with torch.no_grad():
                # visual -> spike
                _, vs_loss = getLoss("V-S", True)
                save("V-S", _)
                # visual -> visual
                _, vv_loss = getLoss("V-V", True)
                save("V-V", _)
                del _
                # spike -> visual
                _, sv_loss = getLoss("S-V", True)
                save("S-V", _)
                # spike -> spike
                _, ss_loss = getLoss("S-S", True)
                save("S-S", _)
                del _
            # Collect info to log
            info = [
                ('V-S', vs_loss), ('S-V', sv_loss),
                ('S-S', ss_loss), ('V-V', vv_loss)
            ]
            # Log to dedicated report file
            ctx.log(ID, '|', ' | '.join([
                f"{t} {v:.04f}" for t, v in info
            ]), file="results.txt", visible=False)
        # Train strategy varies according to TRAIN_MODE
        elif TRAIN_MODE & ALL_MODES:
            def step(loss, optimizer):
                optimizer.zero_grad()
                loss.backward()
                del loss
                optimizer.step()
            # Starting from visual
            _ = None
            if TRAIN_MODE & VIS_ALONE:
                _, vs_loss = getLoss("V-S")
                step(vs_loss, optim_spike)
            if TRAIN_MODE & VIS_JOINT:
                _, vv_loss = getLoss("V-V")  # depends on vs
                step(vv_loss, optim_visual)
            del _, PRED['V-V'], PRED['V-S']
            # Starting from spike
            _ = None
            if TRAIN_MODE & SPI_ALONE:
                _, sv_loss = getLoss("S-V")
                step(sv_loss, optim_visual)
            if TRAIN_MODE & SPI_JOINT:
                _, ss_loss = getLoss("S-S")  # depends on sv
                step(ss_loss, optim_spike)
            del _, PRED['S-S'], PRED['S-V']
        # return super().iterate_batch(ctx, data_point, TRAIN_MODE)
        else:
            assert False, f"Unknown TRAIN_MODE [{TRAIN_MODE}]"
        # Return runtime info to epoch processor
        return LOSS
