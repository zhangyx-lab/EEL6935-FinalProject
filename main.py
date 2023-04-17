#!python
# ---------------------------------------------------------
# Yuxuan Zhang
# Dept. of Electrical and Computer Engineering
# University of Florida
# ---------------------------------------------------------
# Main entry for running train and test on specified model.
# Run this script with no parameters to see help message.
# ---------------------------------------------------------
# Python packages
import random
# PIP Packages
import torch
from torch.utils.data import DataLoader
# User includes
from dataset import DataSet
from lib.Context import Run, RUN_PATH
from lib.Module import Module
from util.loader import train_data, test_data
from util.device import DEVICE
import util.args as args
# Model imports
from models import MODELS
# Initialize datasets
train_set = DataSet(train_data)
test_set = DataSet(test_data)
with Run() as run:
    # Initialize random seed (IMPORTANT!!!)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    # Initialize model
    run.log(banner="Model Initialization")
    Model = MODELS[args.model]
    model: Module = Model(run, DEVICE, train_set.sample())
    model.to(DEVICE)
    # Model too large to be displayed
    run.log(model, file="model.txt", visible=False)
    run.log(model.optimizer, file="optim.txt", visible=False)
    # Record all free parameters
    run.log(banner="Free Parameters")
    run.log("Device        =", DEVICE)
    run.log("Training Mode =", args.train_mode)
    run.log("Num Epochs    =", args.epochs)
    run.log("Batch Size    =", args.batch_size)
    run.log("Random Seed   =", args.seed)
    # Check for previous model to load
    if args.load is not None:
        run.log(banner="Loading States")
        run_list = args.load.split(":")
        run.log(">> Loading model states from:", run_list)
        path_list = [RUN_PATH / _ for _ in run_list]
        model.load(run, *path_list)
    # ================================= TRAIN =================================
    if args.RUN_TRAIN:
        with run.context("train") as ctx:
            train_loader = DataLoader(train_set, batch_size=args.batch_size)
            ctx.log(banner="Training Model")
            model.run(ctx, train_loader, train=args.train_mode)
            ctx.log(banner="Saving States")
            model.save(ctx, run.path)
            # Save model prediction on training set
            ctx.log(banner="Running Prediction on TRAINING SET")
            model.run(ctx, train_set)
    # ================================== TEST =================================
    if args.RUN_TEST:
        with run.context("test") as ctx:
            ctx.log(banner="Running Prediction on TEST SET")
            model.run(ctx, test_set)
    # Congratulations!
    run.log(f"RUN<{run.id}> completed!", banner="Success")
