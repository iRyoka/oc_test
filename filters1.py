import skimage, skimage.morphology
import torch
import numpy as np

def remove_small_holes(img_mask : torch.Tensor, closing_rad : int, holes_threshold : int):
    bg = img_mask[0,:,:].numpy()
    houses = img_mask[2,:,:].numpy()
    forests = img_mask[1,:,:].numpy()
    houses = skimage.morphology.binary_closing(houses, skimage.morphology.disk(closing_rad))
    houses = skimage.morphology.remove_small_holes(houses, 
        area_threshold = holes_threshold)
    forests = skimage.morphology.binary_closing(forests, skimage.morphology.disk(closing_rad))
    forests = skimage.morphology.remove_small_holes(forests, 
        area_threshold = holes_threshold)

    #remove forests where houses are detected
    forests = forests * (1 - houses)

    #remove background where objects are found
    bg = bg * (1 - houses) * (1 - forests)

    newmask = np.stack([bg, forests, houses], axis = 0)

    return torch.tensor(newmask, dtype=torch.bool)

    



