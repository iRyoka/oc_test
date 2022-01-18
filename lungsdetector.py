import skimage, skimage.measure, skimage.morphology
import skimage.feature, skimage.segmentation
import sklearn, sklearn.metrics 
import cv2
import numpy as np
from tqdm.notebook import tqdm

# https://stackoverflow.com/questions/57030125/automatically-adjusting-brightness-of-image-with-opencv
# Automatic brightness and contrast optimization with optional histogram clipping
def automatic_brightness_and_contrast(image, clip_hist_percent=25):
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # CHANNELS_LAST
    gray = image#.reshape(image.shape[0],-1,1)

    # Calculate grayscale histogram
    hist = cv2.calcHist([gray],[0],None,[256],[0,255])
    hist_size = len(hist)
    # print(hist_size)

    # Calculate cumulative distribution from the histogram
    accumulator = []
    accumulator.append(float(hist[0]))
    for index in range(1, hist_size):
        accumulator.append(accumulator[index -1] + float(hist[index]))

    # Locate points to clip
    maximum = accumulator[-1]
    clip_hist_percent *= (maximum/100.0)
    clip_hist_percent /= 2.0

    # print(accumulator)
    # print(clip_hist_percent)

    # Locate left cut
    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1

    # Locate right cut
    maximum_gray = hist_size - 2
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1

    # Calculate alpha and beta values
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha

    auto_result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return (auto_result, alpha, beta)

# converts iage to CL
def channels_last(img : np.ndarray):
    if len(img.shape) == 2:
        return img.reshape(img.shape[0], -1, 1)
    elif len(img.shape) == 3:
        if img.shape[0] <= 3:
            return img.transpose(1,2,0)
    return img

# converts iage to CL
def channels_first(img : np.ndarray):
    if len(img.shape) == 2:
        return img.reshape(1, img.shape[0], -1)
    elif len(img.shape) == 3:
        if img.shape[-1] <= 3:
            return img.transpose(2,0,1)
    return img

# checks if a point lies within thickness from the border of the image
def is_border_point(img, pt, thickness):
    x, y = pt
    h, w = img.squeeze().shape
    return x < thickness or y < thickness or x >= h - thickness or y >= w - thickness

# splits a binary image to connected regions
# if there are less than two, returns None
# otherwise picks two largerst (by area) ones
# and checks if any of them contains a border
# pixel. If so, retuns None. Otherwise returns
# an image with these two regions
def try_to_separate_step(img, border_thickness, verbose = True):
    img = skimage.measure.label(img)
    regions = skimage.measure.regionprops(img)
    regions.sort(key=lambda x: x.area)

    if len(regions) < 2:
        if verbose: print(f"Only {regions} regions found")
        return None

    dst = np.zeros_like(img)
    for regn in regions[-2:]:
        for x,y in regn.coords:
            if is_border_point(img, (x,y), border_thickness):
                if verbose: print("Connected to border")
                return None
            dst[x, y] = 1
    return dst

'''
    Tries to separate lungs from backgroud.
    img                 - CL gray image
    otsu_step           - step at the threshold is decreaed
    black_limit         - 'black' color upper limit on the original image
    clip_percent        - histogram clipping percent for sharpening
    border_thickness    - thickness of a border to avoid
    max_steps           - max number of tries to threshold an image; 
                          should be increased for small otsu_step
    steps_file          - a filename prefix to store all steps to
    verbose             - print fail reason
'''
def separate_lungs(img, otsu_step, black_limit, clip_percent, border_thickness = 3, 
    max_steps = 10, steps_file = None, verbose = False):

    #replace black (< black_limit) with an average of the rest of the image
    black = (img < black_limit).astype(np.uint8)
    black_removed = np.maximum(img, black*np.sum(img)/(np.sum(1-black) + 1e-7)).astype(np.uint8)

    #sharpen the image
    auto_result, _, _ = automatic_brightness_and_contrast(black_removed, clip_percent)

    #find an initial threshold with Otsu's method
    otsu_thresh, otsu_img = cv2.threshold(auto_result, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  

    if steps_file:
        cv2.imwrite(f'{steps_file}_0_src.png', img)
        cv2.imwrite(f'{steps_file}_1_black_removed.png', black_removed)
        cv2.imwrite(f'{steps_file}_2_autocontrast.png', auto_result)
        cv2.imwrite(f'{steps_file}_3_initial_otsu.png', otsu_img)

    imcnt = 3

    sep = try_to_separate_step(otsu_img, border_thickness, verbose)
    while max_steps > 0 and sep is None:
        max_steps -= 1
        otsu_thresh -= otsu_step
        otsu_img = ((auto_result < otsu_thresh) * 255).astype(np.uint8)
        # print(otsu_img)
        sep = try_to_separate_step(otsu_img, border_thickness, verbose)
        if steps_file:
            imcnt += 1
            cv2.imwrite(f'{steps_file}_{imcnt}_another_otsu.png', otsu_img)

    if sep is not None:
        sep = ((sep > 0) * 255).astype(np.uint8)

    if steps_file and sep is not None:
        imcnt += 1
        cv2.imwrite(f'{steps_file}_{imcnt}_unfiltered_result.png', sep)

    return sep

# preforms binary closing with a disk of radius rad
def smoothen(img, rad_closing, rad_dilation=3):
    closing = skimage.morphology.binary_closing(img, skimage.morphology.disk(rad_closing))
    closing = skimage.morphology.binary_dilation(closing, skimage.morphology.disk(rad_dilation))
    return ((closing > 0) * 255).astype(np.uint8)

def F1_single_input(pred_mask, label_mask):
    return sklearn.metrics.f1_score((pred_mask.reshape(-1) > 0).astype(np.uint8), 
            (label_mask.reshape(-1) > 0).astype(np.uint8))

def IOU_single_metrics(pred_mask, label_mask):
    pred_mask = (pred_mask.reshape(-1) > 0).astype(np.uint8)
    label_mask = (label_mask.reshape(-1) > 0).astype(np.uint8)

    intersection = np.sum(pred_mask * label_mask).astype(np.float32)
    union = np.sum(np.maximum(pred_mask, label_mask)).astype(np.float32)

    return intersection / union

def eval_on_dataset(dataset, otsu_step, black_limit, clip_percent, closing_rad = 10, dilation_rad = 3,
        border_thickness = 3, max_steps = 10, out_file = None, verbose = False):
    f1 = 0
    iou = 0
    length = 0
    failed = 0
    for i in tqdm(range(len(dataset)), "Validation: "):
        img, lbl = dataset[i]
        img = channels_last(img)
        lbl = channels_last(lbl)

        pred = separate_lungs(img, otsu_step, black_limit, clip_percent, border_thickness, max_steps, verbose=verbose)

        if pred is None:
            print(f"Can't separate image: {i}")
            failed += 1
            if out_file:
                cv2.imwrite(f'{out_file}_{i}_0_src (FAILED TO CLASSIFY).png', img)
                cv2.imwrite(f'{out_file}_{i}_1_label (FAILED TO CLASSIFY).png', ((lbl > 0) * 255).astype(np.uint8))
            continue

        #Smoothen
        pred = smoothen(pred, closing_rad, dilation_rad) 

        curr_f1 = F1_single_input(pred, lbl)
        curr_iou = IOU_single_metrics(pred, lbl)
        f1 += curr_f1
        iou += curr_iou
        length += 1

        if verbose:
            print(f"Img {i}, F1 {curr_f1}, IOU {curr_iou}")

        if out_file:
            cv2.imwrite(f'{out_file}_{i}_0_src.png', img)
            cv2.imwrite(f'{out_file}_{i}_1_label.png', ((lbl > 0) * 255).astype(np.uint8))
            cv2.imwrite(f'{out_file}_{i}_2_prediction.png', pred)

        #     if (output_filename):
        #         torchvision.io.write_png((img[i,:,:,:].cpu() * 255).to(torch.uint8), 
        #             f"{output_filename}_{j}_1_orig.png")
        #         torchvision.io.write_png((current_pred * 255).to(torch.uint8).unsqueeze(0), 
        #             f"{output_filename}_{j}_2_pred.png")
        #         #overlay
        #         torchvision.io.write_png((current_label * 255).to(torch.uint8).unsqueeze(0), 
        #             f"{output_filename}_{j}_4_label.png")

    return f1/length, iou/length, failed