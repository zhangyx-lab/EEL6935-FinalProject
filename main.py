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
from util.loader import train_data, test_data
from util.run import Run, RUN_PATH
from util.device import DEVICE
from lib.Module import Module
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
    # Record all free parameters
    run.log("Device        =", DEVICE)
    run.log("Num Epochs    =", args.epochs)
    run.log("Batch Size    =", args.batch_size)
    run.log("Optimizer     =", "Adam")
    run.log("Learning Rate =", args.learning_rate)
    run.log("Random Seed   =", args.seed)
    # Initialize model
    run.log(banner="Model Initialization")
    Model: Module = MODELS[args.model]
    model: Module = Model(run, DEVICE, train_set.sample())
    model.to(DEVICE)
    # Model too large to be displayed
    run.log(model, file="model.txt", visible=False)
    # Check for previous model to load
    if args.load is not None:
        load_path = RUN_PATH / args.load
        run.log(">> Loading model states from:", load_path)
        model.load(load_path)
    # ================================= TRAIN =================================
    if args.RUN_TRAIN:
        with run.context("train") as ctx:
            train_loader = DataLoader(train_set, batch_size=args.batch_size)
            ctx.log(banner="Training Model in ALL_ALONE mode")
            model.run(train_loader, ctx, TRAIN_MODE=0b0011)
            ctx.log(banner="Training Model in ALL_JOINT mode")
            model.run(train_loader, ctx, TRAIN_MODE=0b1100)
            ctx.log(banner="Training Model in ALL_ALONE mode")
            model.run(train_loader, ctx, TRAIN_MODE=0b0011)
            ctx.log(banner="Training Model in ALL_MODES")
            model.run(train_loader, ctx, TRAIN_MODE=0b1111)
            # Save model to run dir
            run.log(">> Saving model states to:", run.path)
            model.save(run.path)
            # Save model prediction on training set
            ctx.log(banner="Running Test on TRAINING SET")
            model.run(train_set, ctx)
    # ================================== TEST =================================
    if args.RUN_TEST:
        with run.context("test") as ctx:
            ctx.log(banner="Running Test on TEST SET")
            model.run(test_set, ctx)
    # Congratulations!
    run.log(f"RUN<{run.id}> completed!", banner="Success")
