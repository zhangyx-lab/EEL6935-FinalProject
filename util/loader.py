# ---------------------------------------------------------
# Yuxuan Zhang
# Dept. of Electrical and Computer Engineering
# University of Florida
# ---------------------------------------------------------
# Loads data from local cache and create dataset instances.
# Downloads the data from web if cache could be found.
# ---------------------------------------------------------
import os
import requests
from collections import namedtuple
import numpy as np
import torch
from util.env import DATA_PATH
from cvtb.types import scaleToFit, U8, to_float
import cv2

RES = namedtuple('Resource', ['filename', 'hash'])
KAY_LABELS = RES("kay_labels.npy", "r638s")
KAY_LABELS_VAL = RES("kay_labels_val.npy", "yqb3e")
KAY_IMAGES = RES("kay_images.npz", "ymnjv")


def get_osf_url(hash): return f"https://osf.io/{hash}/download"


def transform(img):
    dtype = img.dtype
    for layer in img:
        # mask = (1 - np.abs(layer * 2 - 1)) ** 2
        blurred = cv2.GaussianBlur(U8(layer), [31, 31], 0)
        blurred = to_float(blurred, dtype)
        layer[:] = blurred

for name, hash in [KAY_LABELS, KAY_LABELS_VAL, KAY_IMAGES]:
    path = str(DATA_PATH / name)
    if not os.path.isfile(path):
        print(f"Downloading {name}...")
        r = requests.get(get_osf_url(hash))
        assert r.status_code == requests.codes.ok, r.status_code
        with open(path, "wb") as f:
            f.write(r.content)
        print(f"Download {name} completed!")



def reorganize(*names, src: np.ndarray, lut: dict[str, int]):
    """
    reorder voxels according to their regions
    """
    ids = [lut[name] for name in names]
    bins = [[] for _ in range(len(names))]
    # Throw indexes into bins
    for i, id in zip(range(len(src)), src):
        if not id in ids:
            print("Warning: throwing away unknown region, id =", id)
        else:
            bins[ids.index(id)].append(i)
    # Collect indexes
    dst = []
    for bin in bins:
        dst += bin
    cnt = [len(bin) for bin in bins]
    # Return index list and count list
    return dst, cnt


# Declare named tuple for dataset storage
Data = namedtuple('DataSet', ['stimuli', 'responses', 'responses_raw', 'labels'])
# Load image dataset as dictionary
with np.load(DATA_PATH / KAY_IMAGES.filename) as dict_obj:
    dat = dict(**dict_obj)
    # ID of Region of Interest
    ROI = dat["roi"]
    # Names of ROI
    ROI_NAMES = dat["roi_names"]
    ROI_INDEXES = {}
    for idx, key in zip(range(len(ROI_NAMES)), ROI_NAMES):
        ROI_INDEXES[key] = idx
    # Mapping of spikes
    __roi_names__ = ["V1", "V2", "V3", "V3A", "V3B", "V4", "LatOcc"]
    VOXEL_MAP, VOXEL_CNT = reorganize(
        *__roi_names__,
        src=ROI, lut=ROI_INDEXES
    )
    # The training set
    train_data = Data(
        # N × 128 × 128 grayscale images (float32)
        stimuli=scaleToFit(dat["stimuli"]),
        # N × 8428 Neural Spike Recordings (float32)
        responses=scaleToFit(dat["responses"][:, VOXEL_MAP]),
        responses_raw=scaleToFit(dat["responses"]),
        # Classification labels predicted by 3rd party models
        labels=np.load(DATA_PATH / KAY_LABELS.filename).T
    )
    transform(train_data.stimuli)
    # The test set
    test_data = Data(
        # N × 128 × 128 grayscale images
        stimuli=scaleToFit(dat["stimuli_test"]),
        # N × 8428 Neural Spike Recordings
        responses=scaleToFit(dat["responses_test"][:, VOXEL_MAP]),
        responses_raw=scaleToFit(dat["responses_test"]),
        # Classification labels predicted by 3rd party models
        labels=np.load(DATA_PATH / KAY_LABELS_VAL.filename).T
    )


__indices__ = []
__total__ = 0
for cnt in VOXEL_CNT:
    __indices__.append([__total__, __total__ + cnt])
    __total__ += cnt


def decompose(spike: torch.Tensor):
    # BatchSize × 8428 Spikes
    assert len(spike.shape) == 2, spike.shape
    return [spike[:, l:r] for l, r in __indices__]

def voxel_count(*names: str):
    result = 0
    for name in names:
        result += VOXEL_CNT[__roi_names__.index(name)]
    return result

if __name__ == "__main__":
    print(ROI_INDEXES)
    print(VOXEL_CNT)
