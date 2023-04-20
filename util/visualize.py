# ---------------------------------------------------------
# Yuxuan Zhang
# Dept. of Electrical and Computer Engineering
# University of Florida
# ---------------------------------------------------------
# TODO: Add description
# ---------------------------------------------------------
import torch
import numpy as np
from math import sqrt, ceil
from cvtb.types import U8, trimToFit


def visual(*vis):
    vis = list(vis)
    for i in range(len(vis)):
        if isinstance(vis[i], torch.Tensor):
            vis[i] = vis[i].detach().cpu().numpy()
    vis = [np.concatenate(list(v), axis=1) for v in vis]
    img = np.concatenate(vis, axis=0)
    return U8(img)


def spike(*spi):
    spi = list(spi)
    for i in range(len(spi)):
        if isinstance(spi[i], torch.Tensor):
            spi[i] = spi[i].detach().cpu().numpy()
    img = []
    for s in spi:
        b, l = s.shape
        w = ceil(sqrt(l))
        s = np.concatenate((s, 0.5 * np.ones((b, w * w - l), s.dtype)), axis=1)
        s = s.reshape((b, w, w)) * 2 - 1
        g = np.ones(s.shape, s.dtype) - np.abs(s)
        bgr = np.stack([1 + s, g, 1 - s], axis=3)
        img.append(bgr)
    img = [np.concatenate(list(i), axis=1) for i in img]
    img = np.concatenate(img, axis=0)
    return U8(trimToFit(img))
