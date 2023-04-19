# ---------------------------------------------------------
# Yuxuan Zhang
# Dept. of Electrical and Computer Engineering
# University of Florida
# ---------------------------------------------------------
# Miscellaneous Components
# ---------------------------------------------------------
import torch
import torch.nn as nn

class PowerActivation(nn.Module):
    def __init__(self, power: float = 0.5):
        super().__init__()
        self.power = power

    def forward(self, x: torch.Tensor):
        sgn = torch.sign(x)
        abs = torch.abs(x)
        return sgn * torch.pow(abs, self.power)


class HiddenLayers(nn.Module):
    def __init__(self, src, dst, delta=0.4, middle=1.2, bias: bool = True):
        super().__init__()
        n = src
        middle = int((src if src > dst else dst) * middle)
        flag_descent = False
        layers = []
        while True:
            if not flag_descent:
                next_n = int(n * (1 + delta))
                layers.append(nn.Linear(n, next_n, bias=bias))
                layers.append(nn.LeakyReLU())
                n = next_n
                if n > middle:
                    flag_descent = True
            else:
                next_n = int(n * (1 - delta))
                if next_n > dst:
                    layers.append(nn.Linear(n, next_n, bias=bias))
                    layers.append(nn.LeakyReLU())
                    n = next_n
                else:
                    layers.append(nn.Linear(n, dst, bias=bias))
                    break
        for layer in layers:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0, std=1e-2)
        self.fc = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        return self.fc(x)
