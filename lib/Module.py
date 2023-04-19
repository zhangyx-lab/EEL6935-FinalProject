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
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
# Local Imports
from lib.Context import Context
from util.env import ensure, exists, Path, relative
import util.args as args
from dataset import DataSet


class Module(torch.nn.Module):

    def __init__(self, device, sample=None, loss=None):
        super().__init__()
        self.device = device
        self.loss = loss

    def save(self, ctx: Context, path: Path):
        """Overload this function for custom loading / saving"""
        name = self.__class__.__name__
        if not isinstance(path, Path):
            path = Path(path)
        # Save model
        model_path = ensure(path / "model") / f"{name}.pkl"
        model = self.state_dict()
        torch.save(model, model_path)
        ctx.log(f"Model state of {name} saved to {relative(model_path)}")
        # Save optimizer
        if hasattr(self, 'optimizer'):
            optim_path = ensure(path / "optim") / f"{name}.pkl"
            optim = self.optimizer.state_dict()
            torch.save(optim, optim_path)
            ctx.log(f"Optim state of {name} saved to {relative(optim_path)}")
        else:
            ctx.log("[WARN] Model", name, "does not have a optimizer")

    def load(self, ctx: Context, *path_list: Path):
        """Overload this function for custom loading / saving"""
        name = self.__class__.__name__
        # Use the first path from the list
        assert len(path_list) == 1, path_list
        # If there is more than 1 elements in the path list,
        # The module should have implemented its own load logic.
        path = path_list[0]
        if not isinstance(path, Path):
            path = Path(path)
        # Load model
        model_path = path / "model" / f"{name}.pkl"
        if exists(model_path):
            model = torch.load(model_path, map_location=self.device)
            self.load_state_dict(model)
            ctx.log(
                f"Model state of {name} loaded from {relative(model_path)}")
        else:
            ctx.log(
                f"[WARNING] {relative(model_path)} does not exist, skipping...")
        # Load optimizer
        optim_path = path / "optim" / f"{name}.pkl"
        if not hasattr(self, 'optimizer'):
            ctx.log("Model", name, "has no optimizer, skipping...")
        elif not exists(optim_path):
            ctx.log("[WARN]", relative(optim_path), "not exist, skipping...")
        else:
            optim = torch.load(optim_path, map_location=self.device)
            self.optimizer.load_state_dict(optim)
            ctx.log(
                f"Optim state of {name} loaded from {relative(optim_path)}")

    # Virtual Function
    def lossFunction(self, pred, truth):
        """Calculates loss from given prediction against ground truth"""
        if self.loss is None:
            print(self.loss)
            raise NotImplementedError
        return self.loss(pred, truth)

    # Virtual Function - Optional
    def iterate_epoch(self, epoch, loader: DataSet, ctx: Context, train=None):
        """
        [ Virtual | Optional ]
        Iterate one epoch
        """
        # Prefix
        prefix = f"Epoch {epoch:4d}" if train else "Test ----"
        # Progress bar
        prog = tqdm(loader, leave=False, ncols=80,
                    bar_format="{l_bar} |{bar}| {n_fmt}/{total_fmt} {rate_fmt}{postfix}")
        prog.set_description(prefix)
        hist = {}
        for data_point in prog:
            if train:
                score = self.iterate_batch(
                    ctx, *data_point, train=train)
            else:
                with torch.no_grad():
                    score = self.iterate_batch(
                        ctx, *data_point, train=train)
            for k in score:
                if k in hist:
                    hist[k] += [score[k]]
                else:
                    hist[k] = [score[k]]
            if ctx.signal.triggered:
                break
        # Generate epoch report
        report = [f"{k} {np.average(hist[k]):.6E}" for k in hist]
        ctx.log(prefix, '|', ' | '.join(report))
        # Save all intermediate data pushed into context during training.
        for key, mem in ctx.collect_all():
            path = ctx.path / key
            res = np.concatenate(mem, axis=0)
            np.save(path, res)
            ctx.log(f"Saved {relative(path)}: {res.shape} ({res.dtype})")

    def iterate_batch(self, ctx: Context, *data_point, train=None):
        """
        [ Virtual | Optional ]
        Iterate one batch
        """
        batch, truth = list(data_point)[:2]
        batch = batch.to(self.device)
        # Forward pass
        prediction = self(batch, train=train)
        # Release batch memory
        del batch
        # Compute truth
        truth = truth.to(self.device)
        # Compute loss
        loss = self.lossFunction(prediction, truth)
        # Check for run mode
        if train:
            # Clear previously computed gradient
            self.optimizer.zero_grad()
            # Backward Propagation
            loss.backward()
            # Optimizing the parameters
            self.optimizer.step()
        else:
            prediction = prediction.detach().cpu().numpy()
            name = self.__class__.__name__
            ctx.push(f"{name}.prediction", prediction)
        # Report results
        return {
            "loss": loss.detach().cpu().numpy()
        }

    def forward(self, x: torch.Tensor, train=None):
        raise NotImplementedError

    def run(self, ctx: Context, loader: DataLoader | DataSet, train=None):
        """
        Run the model in either training mode or testing mode
        """
        # Auto determine number of epochs
        epochs = int(args.epochs) if train else 1
        # Check if loader is torch DataLoader
        if not isinstance(loader, DataLoader):
            assert isinstance(loader, DataSet), loader
            loader = DataLoader(loader, batch_size=1, shuffle=False)
        with ctx.signal:
            for epoch in range(1, epochs + 1):
                self.iterate_epoch(epoch, loader, ctx, train=train)
                if ctx.signal.triggered:
                    break
