# ---------------------------------------------------------
# Yuxuan Zhang
# Dept. of Electrical and Computer Engineering
# University of Florida
# ---------------------------------------------------------
# TODO: Add description
# ---------------------------------------------------------
from torch import Tensor, from_numpy
import torchvision.transforms.functional as f
from random import random
import numpy as np
import cv2


def randfloat(l: float, r: float) -> float:
    if r == l:
        return l
    return l + random() * (r - l)


def affine(*samples: Tensor, rot_r=[-15, 15], trans_r=[-0.1, 0.1], scale_r=[0.8, 1.2], shear=[-10, 10]):
    # -180 ~ +180 degrees
    angle = randfloat(*rot_r)
    # Depending on size of the tensor
    tH, tW = [randfloat(*trans_r) for _ in range(2)]

    def translate(img: Tensor):
        H, W = list(img.shape)[-2:]
        return (int(H * tH), int(W * tW))
    # Any float number
    scale = randfloat(*scale_r)
    # Any float number
    shear = [randfloat(*shear) for _ in range(2)]
    # mapping callback

    def apply(sample: Tensor):
        return f.affine(
            sample,
            angle=angle,
            translate=translate(sample),
            scale=scale,
            shear=shear
        )
    # perform transform on all given samples
    result = list(map(apply, samples))
    # return type depends on length of arglist
    if len(result) == 1:
        result = result[0]
    return result


if __name__ == "__main__":
    x, y = np.mgrid[:250, :250]
    checker = np.logical_xor((x % 100) < 50, (y % 100) < 50)
    checker = np.stack([checker, ~checker, np.ones(
        checker.shape)], axis=2).astype(np.float32)
    img = np.ones((550, 550, 3), checker.dtype)
    img[150:-150, 150:-150] = checker
    cv2.imshow("original", img)
    img = from_numpy(img).swapaxes(0, 2)
    while True:
        a, = affine(img)
        a = a.detach().swapaxes(0, 2).numpy().astype(np.float32)
        cv2.imshow("affine", a)
        k = cv2.waitKey(0)
        if k == ord('q'):
            break
