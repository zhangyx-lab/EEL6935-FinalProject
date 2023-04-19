from .__internal__ import RUN_ID, WORKDIR, args
import numpy as np
import matplotlib.pyplot as plt
from glob import glob


def log2arr(lines: list[str]):
    result = []
    for line in lines:
        segs = line.strip().split('|')
        if len(segs) <= 1:
            continue
        _, value = segs[:2]
        try:
            value = float(value.split()[1])
        except Exception as e:
            print(e)
            print(line)
            break
        result.append(value)
    return result


fig, ax = plt.subplots(
    figsize=(12, 6),
    dpi=300
)

# ax.yaxis.tick_right()
# ax.yaxis.set_label_position("right")

with open(WORKDIR / 'train' / 'log.txt', 'r') as log:
    ax.plot(log2arr(log.readlines())[:-1], 'b-')

fig.savefig(str(WORKDIR / 'vis_train.png'), transparent=True)
