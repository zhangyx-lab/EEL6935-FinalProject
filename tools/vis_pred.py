import cv2
import torch
import numpy as np
from cvtb.UI import Window, Session, KEYCODE
import util.visualize as visualize

def ui(ctx, model, train_set, test_set):
    WIN_VISUAL = Window("Visual View")
    WIN_SPIKE = Window("Spike View")
    count = 0
    with Session(WIN_VISUAL, WIN_SPIKE) as session:
        index = 0
        name = 'test'
        dataset = train_set
        while True:
            if index >= len(dataset):
                index = len(dataset) - 1
            print('index =', index)
            # =====================================================================
            visual, spike, idx = dataset[index]
            visual = torch.stack([visual], dim=0).to(model.device)
            spike = torch.stack([spike], dim=0).to(model.device)
            pred_spike = model.encoder(visual)
            pred_visual = model.decoder(spike)
            # =====================================================================
            vis_visual = visualize.visual(visual, pred_visual)
            WIN_VISUAL.render(vis_visual)
            # =====================================================================
            vis_spike = visualize.spike(spike, pred_spike)
            WIN_SPIKE.render(vis_spike)
            # =====================================================================
            key = session.main_loop()
            if key == ord('q') or key == KEYCODE["escape"]:
                break
            elif key == KEYCODE["enter"] or key == ord('s'):
                cv2.imwrite(str(ctx.path / f"{count}-spike.png"), vis_spike)
                cv2.imwrite(str(ctx.path / f"{count}-visual.png"), vis_visual)
                count += 1
            elif key == KEYCODE["arrow_left"] and index > 0:
                index -= 1
            elif key == KEYCODE["arrow_right"] and index + 1 < len(dataset):
                index += 1
            elif key == KEYCODE["arrow_up"] or key == KEYCODE["arrow_down"]:
                if name == 'test':
                    name = 'train'
                    dataset = train_set
                    print('Viewing training set')
                else:
                    name = 'test'
                    dataset = test_set
                    print('Viewing test set')