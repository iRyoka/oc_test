import os
from typing import Iterable, Union
import pandas as pd
import torchvision
import torchvision.transforms.functional as VISIONF
import torch

class LungsDataset:

    TRANSFORMS_TRAIN = ["randomresizedcrop", "tofloat", "fixlabels"]

    TRANSFORMS_VAL = ["tofloat", "resize", "fixlabels"]

    TRANSFORMS_NUMPY = ["fixlabels", "numpy"]

    def __init__(self, data_dir, img_csv_name, output_size = (128, 128)) -> None:
        with open(os.path.join(data_dir, img_csv_name), "r") as f:
            self.img_csv_data = pd.read_csv(f, delimiter=",")
        self.data_dir = data_dir
        self.transforms = LungsDataset.TRANSFORMS_TRAIN
        self.ouput_size = output_size

    def __len__(self):
        return len(self.img_csv_data.index)

    def train(self): self.transforms = LungsDataset.TRANSFORMS_TRAIN

    def eval(self): self.transforms = LungsDataset.TRANSFORMS_VAL

    def numpy(self): self.transforms = LungsDataset.TRANSFORMS_NUMPY

    def transform(self, img, lbl):
        # print("Lbl shape 1: ", lbl.shape)
        for transform in self.transforms:
            if (transform == "randomresizedcrop"):
                #TODO: magic constants! 
                scale=(0.8, 1.0)
                ratio=(211. / 256., 256. / 211.)

                i, j, h, w = torchvision.transforms.RandomResizedCrop.get_params(img, scale, ratio)

                img = VISIONF.resized_crop(img, i, j, h, w, self.ouput_size)
                lbl = VISIONF.resized_crop(lbl, i, j, h, w, self.ouput_size, 
                    interpolation=VISIONF.InterpolationMode.NEAREST)
            elif (transform == "resize"):
                #print("Lbl shape 2.5: ", lbl.shape)
                img = VISIONF.resize(img, self.ouput_size)
                lbl = VISIONF.resize(lbl, self.ouput_size)
                #print("Lbl shape 3: ", lbl.shape)
            elif (transform == "tofloat"):
                img = torchvision.transforms.ConvertImageDtype(torch.float)(img)
            elif (transform == "fixlabels"): 
                #print("Lbl shape 2: ", lbl.shape)
                lbl = lbl.div(255).squeeze()
            elif (transform == "numpy"):
                img = img.numpy()
                lbl = lbl.numpy()
            else:
                print(f"Warning: unknown transorm ignored: {transform}")

        return img, lbl

    def _img_by_index(self, index : int):
        row = self.img_csv_data.iloc[index]
        impath = row["path_img"]
        lblpath = row["path_msk"]
        img = torchvision.io.read_image(os.path.join(self.data_dir, impath))
        lbl = torchvision.io.read_image(os.path.join(self.data_dir, lblpath))
        return self.transform(img, lbl)

    def __getitem__(self, index : Union[int, slice]):
        if isinstance(index, slice):
            dst = []
            for i in range(len(self))[index]:
                dst.append(self._img_by_index(i))
            return dst
        
        return self._img_by_index(index)



