import torch
from util.args import learning_rate, weight_decay
from torch.optim import Adam

def optimizer(model: torch.nn.Module):
    return Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
