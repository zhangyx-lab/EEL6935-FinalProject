import cv2
import numpy as np
import matplotlib.pyplot as plt
from .__internal__ import WORKDIR, exists, relative
from util.loader import train_data, test_data
from cvtb.UI import Window, Session, KEYCODE
from cvtb.types import scaleToFit, trimToFit

train_pred = {}
test_pred = {}

for store, dirname in zip((train_pred, test_pred), ("train", "test")):
    for key in ["Encoder", "Decoder", "SpikeAE", "VisualAE"]:
        path = WORKDIR / dirname / f"{key}.prediction.npy"
        if exists(path):
            store[key] = np.squeeze(np.load(path))
            print(relative(path), "loaded as", key, *[f(store[key]) for f in (np.min, np.average, np.max)])
        else:
            print(relative(path), "not exist, skipped")


WIN_VISUAL = Window("Visual View")
WIN_SPIKE = Window("Spike View")

with Session(WIN_VISUAL, WIN_SPIKE) as session:
    index = 0
    truth = test_data
    pred =  test_pred
    while True:
        length = len(truth.responses_raw)
        if index >= length:
            index = length - 1
        # =====================================================================
        visual = [truth.stimuli[index]]
        if "Decoder" in pred:
            visual.append(pred["Decoder"][index])
        if "VisualAE" in pred:
            visual.append(pred["VisualAE"][index])
        WIN_VISUAL.render(np.concatenate(
            [scaleToFit(_) for _ in visual], axis=1))
        # =====================================================================
        spike = [truth.responses[index]]
        if "Encoder" in pred:
            spike.append(pred["Encoder"][index])
        if "SpikeAE" in pred:
            spike.append(pred["SpikeAE"][index])
        spike = [scaleToFit(_) for _ in spike]
        def expand(arr):
            arr = arr * 2 - 1
            bgr = trimToFit(np.stack([-arr, arr * 0, arr], axis=1))
            return np.stack([bgr] * 200, axis=0)
        spike = np.concatenate(list(map(expand, spike)), axis=0)
        WIN_SPIKE.render(spike)
        # =====================================================================
        key = session.main_loop()
        if key == ord('q') or key == KEYCODE["escape"]:
            break
        elif key == KEYCODE["arrow_left"] and index > 0:
            index -= 1
        elif key == KEYCODE["arrow_right"] and index + 1 < length:
            index += 1
        # elif key == KEYCODE["arrow_up"] and index > 0:
        #     index -= 1
        # elif key == KEYCODE["arrow_down"] and index + 1 < len:
        #     index -= 1
