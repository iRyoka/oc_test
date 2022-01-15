import os
from typing import Iterable, Iterator, Optional, Union
import pandas as pd
import cv2

class StitchingDataset(Iterable):

    def __init__(self, data_dir, img_csv_name) -> None:
        with open(os.path.join(data_dir, img_csv_name), "r") as f:
            self.img_csv_data = pd.read_csv(f, delimiter=",")
        self.data_dir = data_dir

    def __len__(self):
        return len(self.img_csv_data.index)

    def __iter__(self) -> Iterator:
        return self._image_iterator()

    def _img_by_index(self, index : int):
        row = self.img_csv_data.iloc[index]
        impath = row["path"]
        img = self._img_by_name(os.path.join(self.data_dir, impath))
        return img

    def _img_by_name(self, name : str):
        img = cv2.imread(name)

        #   add alpha-channel for further use as the mask
        #   weirdly, this operation doesn't make all alpha-entries
        #   under pixels equal 255, but only close to it, so that
        #   the values like 254 or 253 are present... 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

        return img

    def __getitem__(self, index : Union[int, slice]):
        if isinstance(index, slice):
            dst = []
            for i in range(len(self))[index]:
                dst.append(self._img_by_index(i))
            return dst
        
        return self._img_by_index(index)

    def _image_iterator(self):
        for (_, val) in self.img_csv_data.iterrows():
            impath = val["path"]
            img = self._img_by_name(os.path.join(self.data_dir, impath))
            yield img


