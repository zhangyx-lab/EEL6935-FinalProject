# ---------------------------------------------------------
# Yuxuan Zhang
# Dept. of Electrical and Computer Engineering
# University of Florida
# ---------------------------------------------------------
# Prints details of the dataset contents and displays
# samples of data points.
# ---------------------------------------------------------
from torch import from_numpy
from util.augment import affine
import termtables as tt
from util.loader import dat, ROI, ROI_NAMES, train_data, test_data
from util.env import VAR_PATH
import numpy as np
import matplotlib.pyplot as plt
from random import shuffle
import cv2
import cvtb

# Report all fields and field types of diction object "dat"
print("\n============ dat ============")
tt.print(
    [[(k, str(dat[k].shape), str(dat[k].dtype)) for k in dat.keys()]],
    header=["key", "shape", "d-type"]
)

# This is the number of voxels in each ROI.
# Note that `"Other"` voxels have been removed from this version of the dataset:
print("\n============ ROI ============")
print("shape", ROI_NAMES)
tt.print(
    list(zip(ROI_NAMES, np.bincount(ROI))),
    header=["ROI Name", "Count"]
)

# [("train_data", train_data), ("test_data", test_data)]:
for set_name, ds in []:
    print(f"\n############ {set_name} ############")
    # `labels` is a 4 by stim array of class names:
    # - row 3 has the labels predicted by a deep neural network (DNN) trained on Imagenet
    # - rows 0-2 correspond to different levels of the wordnet hierarchy for the DNN predictions
    print("\n========== labels v.s. raw images ==========")
    print("Randomly picking from", ds.labels.shape)
    # Generate random indexes to be shown in the table
    indexes = list(range(ds.labels.shape[0]))
    shuffle(indexes)
    indexes = indexes[:4]
    indexes.sort()
    tt.print(
        [[i] + list(ds.labels[i]) for i in indexes],
        header=["", "WordNet P1", "WordNet P2", "WordNet P3", "ImageNet DNN"]
    )
    # Each stimulus is a 128 x 128 grayscale array:
    fig, axs = plt.subplots(1, 4, figsize=(
        8, 4), sharex=True, sharey=True, dpi=300)
    fig.canvas.manager.set_window_title('raw images')
    for ax, i in zip(axs, indexes):
        img = ds.stimuli[i]
        label = ds.labels[i]
        # Render to subplot
        ax.imshow(img, cmap="gray")
        ax.set_title(label[-1])
    fig.tight_layout()
    fig.savefig(str(VAR_PATH / f"{set_name}-visual.png"), transparent=True)

    # Initiate response plot figure
    fig, ax = plt.subplots(figsize=(12, 4), dpi=300)
    fig.canvas.manager.set_window_title('neural recordings')

    # Each stimulus is associated with a pattern of BOLD response across voxels in visual cortex:
    ax.set(xlabel="Voxel", ylabel="Stimulus")
    ax.set_title("responses")
    heatmap = ax.imshow(ds.responses, aspect="auto",
                        vmin=-1, vmax=1, cmap="bwr")
    plt.colorbar(heatmap, shrink=.5, label="Response amplitude (Z)", ax=ax)
    fig.savefig(str(VAR_PATH / f"{set_name}-spike.png"), transparent=True)


def imwrite(path, t: np.ndarray):
    img = cvtb.types.U8(t, cvtb.types.scaleToFit)
    cv2.imwrite(str(path), img)


i = 123

visual = train_data.stimuli[i]
imwrite(VAR_PATH / "sample.visual.png", visual)
visual = from_numpy(visual).reshape((1, 128, 128))
print(visual.shape)
for j in range(6):
    augmented = affine(visual).squeeze().numpy()
    imwrite(VAR_PATH / f"sample.affine-{j}.png", augmented)

fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
ax.set(xlabel="Voxel", ylabel="Amplitude")
spike = train_data.responses[i][:400]
# spike.sort()
ax.bar(range(len(spike)), spike)
fig.savefig(str(VAR_PATH / f"sample.spike.png"), transparent=True)
