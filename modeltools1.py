#https://discuss.pytorch.org/t/is-this-a-correct-implementation-for-focal-loss-in-pytorch/43327/8
# 
# 
from cProfile import label
from pickletools import uint8
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_toolbelt.inference.tiles as ptb_tiles#  ImageSlicer, CudaTileMerger, TileMerger
from torch.utils.data import DataLoader
import numpy as np
from tqdm.notebook import tqdm
import torch.nn.functional as NNF
import torchvision, torchvision.utils, torchvision.transforms
import sklearn, sklearn.metrics
import matplotlib.pyplot as plt
import filters1

DEFAULT_OVERLAY_COLORS = ["gray", "green","orange"]

class FocalLoss(nn.Module):
    
    def __init__(self, weight=None, 
                 gamma=2., reduction='none'):
        nn.Module.__init__(self)
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, input_tensor, target_tensor):
        log_prob = F.log_softmax(input_tensor, dim=1)
        prob = torch.exp(log_prob)
        return F.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob, 
            target_tensor, 
            weight=self.weight,
            reduction = self.reduction
        )

'''
    Provides evaluation on large images via torch toolbelt's tiler
    
    model           - torch model to apply. It is forcibly switched to eval mode.
    image           - image in torchvision float format, channels first
    batch_size      - size of a batch of tiles to process at once
    tile_size       - size of a tile (pair)
    tile_step       - step of tiling. Recommended value: tile_size/2
    model_device    - a device to send the model and inputs to
    merger_device   - a device to merge the resulting image on. 
                    Both device to "cuda" is faster but uses
                    more VRAM

    Output is an np.ndarray, channels first
'''
def tiled_eval(model : torch.nn.Module, image, batch_size, 
        tile_size : Tuple[int, int], tile_step : Tuple[int, int], 
        model_device, merger_device) -> np.ndarray:
    image = image.numpy().transpose(1,2,0)
    model.eval()
    model.to(model_device)
    tiler = ptb_tiles.ImageSlicer(image.shape, 
        tile_size=tile_size, tile_step=tile_step)
    tiles = tiler.split(image)
    # Allocate a CUDA buffer for holding entire mask
    merger = ptb_tiles.TileMerger(tiler.target_shape, 3, tiler.weight, device = merger_device)

    # Run predictions for tiles and accumulate them
    for tiles_batch, coords_batch in tqdm(DataLoader(list(zip(tiles, tiler.crops)), 
            batch_size=batch_size, 
            num_workers=1, pin_memory=True)):
        tiles_batch = tiles_batch.to(model_device).permute(0,3,1,2)
        with torch.no_grad():
            pred_batch = model(tiles_batch)#.to(merger_device)
        merger.integrate_batch(pred_batch, coords_batch)

    result = merger.merge().cpu().numpy().transpose(1,2,0)
    cropped = tiler.crop_to_orignal_size(result).transpose(2,0,1)

    return cropped

def classes_to_masks(img, num_classes = 3):
    img = img.to(torch.int64)
    prediction_masks = NNF.one_hot(img, num_classes = num_classes).permute(2, 0, 1).to(torch.bool)
    return prediction_masks

def logits_to_masks(inputimg, num_classes = 3):
    #convert to tensor
    if isinstance(inputimg, np.ndarray):
        inputimg = torch.tensor(inputimg)

    #convert to channels first
    if inputimg.size(0) != num_classes:
        if inputimg.size(2) == num_classes:
            inputimg = inputimg.permute(2, 0, 1)
        else:
            raise ValueError("input must be 3-d with either CF or CL")

    # print(inputimg.shape)

    labels = torch.argmax(inputimg, dim = 0).to(torch.int64)

    # print(labels)
    # prediction_masks = NNF.one_hot(labels, num_classes = num_classes).permute(2, 0, 1).to(torch.bool)
    return classes_to_masks(labels, num_classes)

# input: torch, CF
# color can be None
def apply_mask(img, masks, colors, alpha = 0.3):
    img = torchvision.transforms.ConvertImageDtype(torch.uint8)(img)
    masked = torchvision.utils.draw_segmentation_masks(img, masks, colors=colors, alpha=0.3)
    return masked

# perdictions must be in mask format; channels first
def F1_score(prediction, label, class_weights = []):
    if not class_weights:
        class_weights = [1] * prediction.shape[0]
    total_score = 0
    for channel, weight in zip(range(prediction.shape[0]), class_weights):
        pred = (prediction[channel,:,:].numpy() > 0).astype(np.uint8).reshape(-1)
        lab = (label[channel,:,:].numpy() > 0).astype(np.uint8).reshape(-1)
        score = sklearn.metrics.f1_score(pred, lab) * weight
        total_score += score
    return total_score / sum(class_weights)

#no extension 
def eval_on_dataset(model, dataset, output_filename, tile_batch_size, tile_size, 
        tile_step, model_device, merger_device, class_weights = [], 
        apply_filter = False, filter_rad = None, filter_area = None):
    scores = []
    for i in tqdm(range(len(dataset)), "Images in dataset: "):
        img, lbl = dataset[i]
        prediction = tiled_eval(model, img, tile_batch_size, tile_size, tile_step, model_device, merger_device)
        pred_mask = logits_to_masks(prediction)
        label_mask = classes_to_masks(lbl)
        if apply_filter:
            pred_mask_filtered = filters1.remove_small_holes(pred_mask, 
            filter_rad, filter_area)
        else:
            pred_mask_filtered = pred_mask
        score = F1_score(pred_mask_filtered, label_mask, class_weights)
        print(f"Score {i}:", score)
        scores.append(score)
        if output_filename:
            #compute overlay
            overlayed_pred = apply_mask(img, pred_mask, DEFAULT_OVERLAY_COLORS, alpha = 0.4)
            overlayed_label = apply_mask(img, label_mask, DEFAULT_OVERLAY_COLORS, alpha = 0.4)
            torchvision.io.write_png((img * 255).to(torch.uint8), f"{output_filename}_{i}_1_orig.png")
            torchvision.io.write_png(((pred_mask > 0) * 127).to(torch.uint8), f"{output_filename}_{i}_2_mask.png")
            torchvision.io.write_png(overlayed_pred, f"{output_filename}_{i}_3_overlay.png")
            torchvision.io.write_png(((label_mask > 0) * 127).to(torch.uint8), f"{output_filename}_{i}_4_mask_GT.png")
            torchvision.io.write_png(overlayed_label, f"{output_filename}_{i}_5_mask_GT_overlay.png")
            if apply_filter:
                overlayed_filtered_label = apply_mask(img, pred_mask_filtered, DEFAULT_OVERLAY_COLORS, alpha = 0.4)
                torchvision.io.write_png(((pred_mask_filtered > 0) * 127).to(torch.uint8), f"{output_filename}_{i}_6_filtered_mask.png")
                torchvision.io.write_png(overlayed_filtered_label, f"{output_filename}_{i}_7_filtered_overlay.png")



    score = np.mean(np.array(scores))
    return score

def train_model(model, loss, optimizer, scheduler, dataloader, num_epochs, device, 
        print_epoch_summary = True, plot_losses = True):
    model.train()
    model.to(device)
    losses = []
    for i in tqdm(range(num_epochs)):
        # print("Epoch: ", i)
        for batchnum, batch in enumerate(dataloader):
            # print("Batchnum: ", batchnum)
            optimizer.zero_grad()

            imgs, labels = batch
            imgs = imgs.to(device)
            labels = labels.to(device)
            dst = model(imgs)

            lossval = loss(dst.to(torch.float), labels.to(torch.long))
            losscpu = lossval.detach().cpu().numpy()
            losses.append(losscpu)
            lossval.backward()

            optimizer.step()
            scheduler.step()

        if print_epoch_summary:
            print(f"Epoch {i}, batch {batchnum}, loss: {losscpu}, lr: {scheduler.get_last_lr()}")
    if plot_losses:                
        plt.plot(range(len(losses)), losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')