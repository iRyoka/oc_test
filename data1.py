from doctest import NORMALIZE_WHITESPACE
import os
from typing import Tuple
import numpy as np
import pandas as pd
import torch, torch.nn
import torch.nn.functional as NNF
import torchvision, torchvision.utils, torchvision.transforms
from torchvision.io.image import ImageReadMode
import torchvision.transforms.functional as VISIONF
from tqdm import tqdm as tqdm
import matplotlib.pyplot as plt
import math
from tqdm.notebook import tqdm as tqdm
import kornia, kornia.color

DEFAULT_OVERLAY_COLORS = ["gray", "green","orange"]

class Dataset(torch.utils.data.Dataset):

    TRAIN_TRANSFORMS = ["randomresizedcrop", "jitter", 
	    "tofloat", "normalize", "randomflip", 
        "fixlabels"]
    VAL_TRANSFORMS = ["tofloat", "fixlabels", "resize_src_to_labels", "normalize"]

    VAL_TRANSFORMS_NO_NORM = ["tofloat", "fixlabels", "resize_src_to_labels"]

    """
        transforms may contain the following ones:
            "randomresizedcrop" -- RandomResizedCrop
            "tofloat" -- change src dtype to float (leaves label as is)
            "onehotlabels" -- change label type to class-wise one-hot
    """
    def __init__(self, data_dir : str, data_csv : str,  
        output_size : Tuple[int,int] = (128, 128)) -> None:
        super().__init__()

        self.transforms = Dataset.TRAIN_TRANSFORMS
        self.output_size = output_size
        #self.use_transforms = True
        self.jitter = torchvision.transforms.ColorJitter(0.4, 0.4, 0.4, 0.4)

        train_datafile = pd.read_csv(os.path.join(data_dir, data_csv), delimiter=",")

        self.train_images = []

        #the dataset is relatively small, so we will store it memory to
        #avoid HDD latencies
        for _, row in tqdm(train_datafile.iterrows(), "Loading raw data"):
            img_src = torchvision.io.read_image(os.path.join(data_dir,row["path_img"]))
            img_mask = torchvision.io.read_image(os.path.join(data_dir,row["path_msk"]), 
                ImageReadMode.GRAY)
            self.train_images.append([img_src, img_mask])

    def train(self):
        self.transforms = Dataset.TRAIN_TRANSFORMS
    
    def eval(self):
        self.transforms = Dataset.VAL_TRANSFORMS
    
    def eval_no_norm(self):
        self.transforms = Dataset.VAL_TRANSFORMS_NO_NORM

    def __len__(self):
        return len(self.train_images)

    def __getitem__(self, index):
        img, label = self.train_images[index]

        for transform in self.transforms:
            if (transform == "randomresizedcrop"):
                #TODO: magic constants! 
                scale=(0.08, 1.3)
                ratio=(3. / 4., 4. / 3.)

                i, j, h, w = torchvision.transforms.RandomResizedCrop.get_params(img, scale, ratio)

                img = VISIONF.resized_crop(img, i, j, h, w, self.output_size)
                label = VISIONF.resized_crop(label, i, j, h, w, self.output_size, 
                    interpolation=VISIONF.InterpolationMode.NEAREST)
            #hotfix for problems with validation dataset
            elif (transform == "resize_src_to_labels"):
                if img.shape[1:] != label.shape[1:]:
                    img = VISIONF.resize(img, label.shape)
            elif (transform == "tofloat"):
                img = torchvision.transforms.ConvertImageDtype(torch.float)(img)
            elif (transform == "fixlabels"): 
                label = label.div(127, rounding_mode="floor").squeeze()
            elif (transform == "randomflip"):
                if torch.rand(1) < 0.5:
                    img = VISIONF.vflip(img)         
                    label = VISIONF.vflip(label)         
                if torch.rand(1) < 0.5:
                    img = VISIONF.hflip(img)   
                    label = VISIONF.hflip(label)               
            elif (transform == "jitter"):
                img = self.jitter(img)
            elif (transform == "normalize"):
                mean_rgb=(0.4914, 0.4822, 0.4465)
                std_rgb=(0.2023, 0.1994, 0.2010)
                img = VISIONF.normalize(kornia.color.hsv_to_rgb(img), mean_rgb, std_rgb)
            elif (transform == "normalize_rgb"):
                mean_rgb=(0.4914, 0.4822, 0.4465)
                std_rgb=(0.2023, 0.1994, 0.2010)
                img = VISIONF.normalize(img, mean_rgb, std_rgb)
            else:
                print(f"Warning: unknown transform ignored: {transform}")

        return img, label

def show(imgs, labels=[]):
    if not isinstance(imgs, list):
        imgs = [imgs]
    plt.figure(dpi=60)
    _, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = torchvision.transforms.functional.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        if labels:
            axs[0,i].set_title(labels[i])
        plt.savefig('filename.png', dpi=1200)

def draw_prediction(src, label, model, colors = DEFAULT_OVERLAY_COLORS, 
        device = torch.device("cpu"), precomputed_prediction_classes = None):
    prediction_class = precomputed_prediction_classes
    if prediction_class is None:        
        src_cuda = src
        if len(src.shape) == 3: src_cuda = src.unsqueeze(0)

        src_cuda = src_cuda.to(device)

        #print(device)
        #print(src_cuda.device)
        prediction = model(src_cuda).detach()
        prediction_class = torch.argmax(prediction, dim = 1).to(torch.int64)
    
    src = torchvision.transforms.ConvertImageDtype(torch.uint8)(src)

    prediction_masks = NNF.one_hot(prediction_class, num_classes = 3)
    if prediction_masks.dim() == 3: 
        prediction_masks = prediction_masks.unsqueeze(0)
    print(f"Prediction mask shape: {prediction_masks.shape}")
    prediction_masks = prediction_masks.transpose(0, 3).to(torch.bool).squeeze()

    # print(src.shape, prediction_class.shape, prediction_masks.shape, label.shape)
    dst = torchvision.utils.draw_segmentation_masks(src, prediction_masks,
        colors=colors, alpha=0.3)
    #print(prediction_class)
    show([src, label, prediction_class.squeeze().to(torch.uint8), dst],
        ["src", "label", "predicted", "overlay"])

# def predict_by_patches(src, tile_size : Tuple[int, int], 
#         model : torch.nn.Module, use_softmax : bool,
#         device = "cpu"):
#     h, w = src.shape[-2:]
#     tileh, tilew = tile_size
#     assert tilew % 2 == 0 and tileh %2 == 0, "so far we only support even tile sizes"
#     pad_left, pad_right, pad_top, pad_bottom = 0, 0, 0, 0
#     if w % tilew != 0:
#         #pad horizontally
#         padw = tilew * int(math.ceil(w / tilew)) - w
#         pad_left = padw // 2
#         pad_right = padw - pad_left
#     if h % tileh != 0:
#         #pad vertically
#         padh = tileh * int(math.ceil(h / tileh)) - h
#         pad_top = padh // 2
#         pad_bottom = padh - pad_top
#     padded_src = torchvision.transforms.Pad([pad_left, pad_top, pad_right, pad_bottom], padding_mode="reflect")(src)
#     paddedw = w + pad_left + pad_right
#     paddedh = h + pad_top + pad_bottom
    
#     result = torch.zeros_like(padded_src)

#     #number of steps of considering a tile
#     wsteps = 2 * (paddedw // tilew) - 1
#     hsteps = 2 * (paddedh // tileh) - 1

#     print(f"Padding input: {src.shape}, output: {padded_src.shape}, wsteps: {wsteps}, hsteps: {hsteps}")
#     # return
    
#     half_tileh = tileh // 2
#     half_tilew = tilew // 2

#     tile = None #TODO:????
#     tile_prediction = None

#     for wpos in tqdm(range(wsteps)):
#         for hpos in range(hsteps):
#             tile = padded_src[:,hpos * half_tileh : hpos * half_tileh + tileh, wpos * half_tilew : wpos * half_tilew + tilew]
#             # print("Iteration start")
#             #print(f"Tile shape: {tile.shape}")
#             #torch.cuda.empty_cache() 
#             tile_prediction = model(tile.unsqueeze(0).to(device)).cpu().squeeze()
#             #TODO:should we use raw logits?
#             # print(tile.shape, tile_prediction.shape)
#             if use_softmax:
#                 tile_prediction = NNF.softmax(tile_prediction, dim = 0)
#             # print(tile_prediction.shape)
#             result[:,hpos * half_tileh : hpos * half_tileh + tileh, wpos * half_tilew : wpos * half_tilew + tilew] = \
#                 result[:,hpos * half_tileh : hpos * half_tileh + tileh, wpos * half_tilew : wpos * half_tilew + tilew] + \
#                 tile_prediction

#     result = torch.argmax(result, dim = 0)
#     print(f"PRE-crop result shape {result.shape}")

#     #crop to original size
#     result = result.squeeze()[pad_top:pad_top+h,pad_left:pad_left+w]
#     print(f"POST-crop result shape {result.shape}")

#     return result

