# ---------------------------------------------------------
# Yuxuan Zhang
# Dept. of Electrical and Computer Engineering
# University of Florida
# ---------------------------------------------------------
# U_Net Node Module
# ---------------------------------------------------------
import torch.nn as nn

class Node(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(in_ch)
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm2 = nn.BatchNorm2d(out_ch)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        # Random init
        for layer in [self.conv1, self.conv2]:
            nn.init.normal_(layer.weight, mean=0, std=1e-2)

    def forward(self, x, train=False):
        if train:
            x = self.norm1(x)
        x = self.relu1(x)
        x = self.conv1(x)
        if train:
            x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv2(x)
        return x
