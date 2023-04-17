from os.path import isdir
import matplotlib.pyplot as plt
from util.env import VAR_PATH, RUN_PATH, Path, exists
from glob import glob

plt.rcParams.update({
    "lines.color": "white",
    "patch.edgecolor": "white",
    "text.color": "black",
    "axes.facecolor": "white",
    "axes.edgecolor": "lightgray",
    "axes.labelcolor": "white",
    "axes.titlecolor": "white",
    "xtick.color": "white",
    "ytick.color": "white",
    "grid.color": "lightgray",
    "figure.facecolor": "black",
    "figure.edgecolor": "black",
    "savefig.facecolor": "black",
    "savefig.edgecolor": "black"})


def log2arr(lines: list[str]):
    result = []
    for line in lines:
        line = line.strip()
        if not line.startswith('Epoch'):
            continue
        _, value = line.split('|')[:2]
        # epoch = int(epoch.split()[1])
        value = float(value.split()[1])
        result.append(value)
    return result


subplots: list[tuple[str, list]] = []

for path in glob(str(RUN_PATH / "*")):
    path = Path(path)
    runID = str(path.relative_to(RUN_PATH))
    if not isdir(path):
        continue
    if exists(path / "000_SUCCESS"):
        continue
    log_path = path / "train" / "log.txt"
    if not exists(log_path):
        continue
    # Load progress from log
    with open(log_path, 'r') as log:
        subplots.append((runID, log2arr(log.readlines())))


fig, axs = plt.subplots(
    len(subplots), 1,
    figsize=(12, len(subplots) * 2),
    dpi=300
)

axs = list(axs)[:len(subplots)]

fig.tight_layout(h_pad=3)

for (runID, line), ax in zip(subplots, axs):
    ax.set_title(runID)
    ax.plot(line, 'r-')

fig.savefig(str(VAR_PATH / 'progress.png'), transparent=True)
