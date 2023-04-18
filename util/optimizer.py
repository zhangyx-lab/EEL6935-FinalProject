# ---------------------------------------------------------
# Yuxuan Zhang
# Dept. of Electrical and Computer Engineering
# University of Florida
# ---------------------------------------------------------
# TODO: Add description
# ---------------------------------------------------------
import torch
from util.args import learning_rate, weight_decay
from torch.optim import Adam

def optimizer(model: torch.nn.Module):
    return Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
