# ---------------------------------------------------------
# Yuxuan Zhang
# Dept. of Electrical and Computer Engineering
# University of Florida
# ---------------------------------------------------------
# Prints details of the dataset contents and displays
# samples of data points.
# ---------------------------------------------------------
import termtables as tt
from util.loader import dat, ROI, ROI_NAMES, train_data, test_data
from util.env import VAR_PATH
import numpy as np
import matplotlib.pyplot as plt
from random import shuffle

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

for set_name, ds in [("train_data", train_data), ("test_data", test_data)]:
    print(f"\n############ {set_name} ############")
    # `labels` is a 4 by stim array of class names:
    # - row 3 has the labels predicted by a deep neural network (DNN) trained on Imagenet
    # - rows 0-2 correspond to different levels of the wordnet hierarchy for the DNN predictions
    print("\n========== labels v.s. raw images ==========")
    print("Randomly picking from", ds.labels.shape)
    # Generate random indexes to be shown in the table
    indexes = list(range(ds.labels.shape[1]))
    shuffle(indexes)
    indexes = indexes[:8]
    indexes.sort()
    tt.print(
        [[i] + list(ds.labels[:4, i]) for i in indexes],
        header=["", "WordNet P1", "WordNet P2", "WordNet P3", "ImageNet DNN"]
    )
    # Each stimulus is a 128 x 128 grayscale array:
    fig, axs = plt.subplots(2, 4, figsize=(12, 6), sharex=True, sharey=True, dpi=600)
    fig.canvas.manager.set_window_title('raw images')
    for ax, i in zip(axs.flat, indexes):
        img = ds.stimuli[i]
        print(img.shape)
        label = ds.labels[:, i]
        # Render to subplot
        ax.imshow(img, cmap="gray")
        ax.set_title(label[-1])
    fig.tight_layout()
    fig.savefig(str(VAR_PATH / f"{set_name}-visual.png"), transparent=True)

    # Initiate response plot figure
    fig, ax = plt.subplots(figsize=(12, 5), dpi=600)
    fig.canvas.manager.set_window_title('neural recordings')

    # Each stimulus is associated with a pattern of BOLD response across voxels in visual cortex:
    ax.set(xlabel="Voxel", ylabel="Stimulus")
    ax.set_title("responses")
    heatmap = ax.imshow(ds.responses, aspect="auto", vmin=-1, vmax=1, cmap="bwr")
    plt.colorbar(heatmap, shrink=.5, label="Response amplitude (Z)", ax=ax)
    fig.savefig(str(VAR_PATH / f"{set_name}-spike.png"), transparent=True)
