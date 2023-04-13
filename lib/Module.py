# ---------------------------------------------------------
# Yuxuan Zhang
# Dept. of Electrical and Computer Engineering
# University of Florida
# ---------------------------------------------------------
# Prototype of the custom Module class based on the pytorch
# nn.Module
# ---------------------------------------------------------
# External Packages
import torch
import numpy as np
from tqdm import tqdm
# Local Imports
from lib.Signal import Signal
from util.run import Context
import util.args as args
from dataset import DataSet


class Module(torch.nn.Module):

    def __init__(self, device, sample=None):
        super().__init__()
        self.device = device

    def save(self, path):
        # Save model
        model = self.state_dict()
        torch.save(model, path / "model.pkl")
        # Save optimizer
        optim = self.optimizer.state_dict()
        torch.save(optim, path / "optim.pkl")

    def load(self, path):
        # Load model
        model = torch.load(str(path / "model.pkl"))
        self.load_state_dict(model)
        # Load optimizer
        optim = torch.save(optim, path / "optim.pkl")
        self.optimizer.load_state_dict(optim)

    # Virtual Function
    def lossFunction(self, pred, truth):
        """Calculates loss from given prediction against ground truth"""
        raise NotImplementedError

    # Virtual Function - Optional
    def score(self, pred, truth, loss):
        """
        [ Virtual | Optional ]
        Calculates the score based on given prediction, ground truth and loss
        """
        return [("loss", loss), ("anything", 0)]

    # Virtual Function - Optional
    def iterate_epoch(self, epoch, loader: DataSet, ctx: Context, TRAIN_MODE=False):
        """
        [ Virtual | Optional ]
        Iterate one epoch
        """
        # Prefix
        prefix = f"Epoch {epoch:03d} |" if TRAIN_MODE else "Test ---- |"
        # Progress bar
        prog = tqdm(loader, leave=False)
        prog.set_description(prefix)
        score_sum = None
        count = 0
        for data_point in prog:
            count += 1
            if TRAIN_MODE:
                score = self.iterate_batch(ctx, data_point, TRAIN_MODE=TRAIN_MODE)
            else:
                with torch.no_grad:
                    score = self.iterate_batch(
                        ctx, data_point, TRAIN_MODE=TRAIN_MODE)
            if score_sum is None:
                score_sum = score
            else:
                for k in score:
                    assert k in score_sum, k
                    score_sum[k] += score[k]
        ctx.log(
            prefix,
            ' | '.join([f"{k} {score_sum[k] / count:.4f}" for k in score_sum])
        )

    # Virtual Function - Optional

    def iterate_batch(self, ctx: Context, data_point: list[torch.Tensor, torch.Tensor, torch.Tensor], TRAIN_MODE=False):
        """
        [ Virtual | Optional ]
        Iterate one batch
        """
        batch, truth, name = data_point
        batch = batch.to(self.device)
        # Forward pass
        prediction = self(batch, train=TRAIN_MODE)
        # Release batch memory
        del batch
        # Compute truth
        truth = truth.to(self.device)
        # Compute loss
        loss = self.lossFunction(prediction, truth)
        # Check for run mode
        if TRAIN_MODE:
            # Clear previously computed gradient
            self.optimizer.zero_grad()
            # Backward Propagation
            loss.backward()
            # Optimizing the parameters
            self.optimizer.step()
        else:
            np.save(ctx.path / str(name), prediction)
        # Report results
        return self.score(prediction, truth, loss)

    # Run the model in either training mode or testing mode
    def run(self, loader: DataSet, ctx: Context, TRAIN_MODE=False):
        epochs = int(args.epochs) if TRAIN_MODE else 1
        with Signal(context=ctx) as sig:
            for epoch in range(1, epochs + 1):
                self.iterate_epoch(epoch, loader, ctx, TRAIN_MODE=TRAIN_MODE)
                if sig.triggered:
                    break
