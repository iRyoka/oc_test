import torch
import torchvision, torchvision.io
import sklearn, sklearn.metrics
from tqdm.notebook import tqdm
import numpy as np


def F1_single_input(pred_mask, label_mask):
    return sklearn.metrics.f1_score((pred_mask.numpy().reshape(-1) > 0).astype(np.uint8), 
            (label_mask.numpy().reshape(-1) > 0).astype(np.uint8))

def IOU_single_metrics(pred_mask, label_mask):
    pred_mask = (pred_mask.numpy().reshape(-1) > 0).astype(np.uint8)
    label_mask = (label_mask.numpy().reshape(-1) > 0).astype(np.uint8)

    intersection = np.sum(pred_mask * label_mask).astype(np.float32)
    union = np.sum(np.maximum(pred_mask, label_mask)).astype(np.float32)

    return intersection / union

def eval_model_on_dataloader(model, dataloader, device, output_filename):
    f1 = 0
    iou = 0
    length = 0
    j = 0
    with torch.no_grad():
        for img, lbl in tqdm(dataloader, "Validation: "):
            img.to(device)
            logits = model(img.to(device))
            # print("Logits", logits.shape)
            pred = torch.argmax(logits, dim = 1).cpu()
            for i in range(img.shape[0]):
                j += 1
                current_pred = pred[i,:,:]
                current_label = lbl[i,:,:]
                f1 += F1_single_input(current_pred, current_label)
                iou += IOU_single_metrics(current_pred, current_label)
                length += 1
                if (output_filename):
                    torchvision.io.write_png((img[i,:,:,:].cpu() * 255).to(torch.uint8), 
                        f"{output_filename}_{j}_1_orig.png")
                    torchvision.io.write_png((current_pred * 255).to(torch.uint8).unsqueeze(0), 
                        f"{output_filename}_{j}_2_pred.png")
                    #overlay
                    torchvision.io.write_png((current_label * 255).to(torch.uint8).unsqueeze(0), 
                        f"{output_filename}_{j}_4_label.png")

    return f1/length, iou/length