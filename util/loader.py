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
from util.env import DATA_PATH

RES = namedtuple('Resource', ['filename', 'hash'])
KAY_LABELS = RES("kay_labels.npy", "r638s")
KAY_LABELS_VAL = RES("kay_labels_val.npy", "yqb3e")
KAY_IMAGES = RES("kay_images.npz", "ymnjv")


def get_osf_url(hash): return f"https://osf.io/{hash}/download"


for name, hash in [KAY_LABELS, KAY_LABELS_VAL, KAY_IMAGES]:
    path = str(DATA_PATH / name)
    if not os.path.isfile(path):
        print(f"Downloading {name}...")
        r = requests.get(get_osf_url(hash))
        assert r.status_code == requests.codes.ok, r.status_code
        with open(path, "wb") as f:
            f.write(r.content)
        print(f"Download {name} completed!")
# Declare named tuple for dataset storage
Data = namedtuple('DataSet', ['stimuli', 'responses', 'labels'])
# Load image dataset as dictionary
with np.load(DATA_PATH / KAY_IMAGES.filename) as dict_obj:
    dat = dict(**dict_obj)
    # The training set
    train_data = Data(
        # N × 128 × 128 grayscale images (float32, [0 ~ 1])
        stimuli=dat["stimuli"],
        # N × 8428 Neural Spike Recordings (float32, [-1 ~ 1])
        responses=dat["responses"],
        # Classification labels predicted by 3rd party models
        labels=np.load(DATA_PATH / KAY_LABELS.filename).T
    )
    # The test set
    test_data = Data(
        # N × 128 × 128 grayscale images
        stimuli=dat["stimuli_test"],
        # N × 8428 Neural Spike Recordings
        responses=dat["responses_test"],
        # Classification labels predicted by 3rd party models
        labels=np.load(DATA_PATH / KAY_LABELS_VAL.filename).T
    )
    # ID of Region of Interest
    ROI = dat["roi"]
    # Names of ROI
    ROI_NAMES = dat["roi_names"]
