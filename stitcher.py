from enum import IntEnum
from typing import Iterable
import cv2
import numpy as np
from tqdm.notebook import tqdm

class GainCorrection(IntEnum):
    NONE = 0
    LOCAL = 1
    LOCAL_AND_GLOBAL = 2

class Stitcher:

    def __init__(self, SIFT_features_limit = 10000):
        self.SIFT_features_limit = SIFT_features_limit
        self.detector = cv2.SIFT_create(nfeatures = SIFT_features_limit)

        #default params
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

        self._gain_vals = None
        self._gain_measurements = None

    """
        Looks for a homography based on provided keypoints and SIFT descriptors
    """
    def getHomography(self, keypoints1, descriptors1, keypoints2, descriptors2) -> np.ndarray:

        matches = self.flann.knnMatch(descriptors1, descriptors2, k=2) #top 2 mathces to pass through filter

        #filter matches that are far apart
        good_matches = []
        for m,n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)

        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches ]).reshape(-1,1,2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches ]).reshape(-1,1,2)
        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        return H
    
    """
        Calculates the ROI within the accumulate and pasted images
        shift is a 2d-vector that describes the position of the new 
        image on the original one
    """
    def getROIboxes(self, dst_size, src_size, shift):
        shift = np.array([shift[1], shift[0]])
        dst_w, dst_h = dst_size
        src_w, src_h = src_size + shift

        # print(f"Shift: {shift}")
        # print(f"Dst size: {dst_size}")
        # print(f"src size: {src_size}")
        # print(f"Src + shift size: {src_size + shift}")

        bb_orig = (max(0, shift[0]), min(dst_w, src_w), max(0, shift[1]), min(dst_h, src_h))
        bb_new = (bb_orig[0] - shift[0], bb_orig[1] - shift[0], 
                bb_orig[2] - shift[1], bb_orig[3] - shift[1])

        # print(f"bb_orig: {bb_orig}")
        # print(f"bb_new: {bb_new}")    

        return bb_orig, bb_new

    #inplace gain correction
    def _apply_gain(self, img, gain):
        gain = gain.reshape(1,1,-1)
        #TODO: experimental
        gain = np.mean(gain) * np.ones_like(gain)

        img[:,:,:3] = (np.minimum(img[:,:,:3].astype(np.float32) * gain, 
                255 * np.ones_like(img[:,:,:3]))).astype(np.uint8)

    '''
        warning: inplace!
        Blends new image onto the original one treating alpha-channel
        as a mask. If a pixel is present on both images, they are blended
        with a coefficient 0.5 
        The new mask is stored to the alpha-channel of the result.
        Provides gain correction if requested
    '''
    def binAlphaBlend(self, img_orig, img_new, shift, gain_correction : bool, compute_mse : bool):

        bb_orig, bb_new = self.getROIboxes(img_orig.shape[:2], img_new.shape[:2], shift)
        orig_left, orig_right, orig_top, orig_bottom = bb_orig
        new_left, new_right, new_top, new_bottom = bb_new
        slice_orig = np.s_[orig_left:orig_right,orig_top:orig_bottom,:]
        slice_new = np.s_[new_left:new_right,new_top:new_bottom,:]

        # print("Slices: ")
        # print(slice_orig)
        # print(slice_new)

        ROI_orig = img_orig[slice_orig]
        ROI_new = img_new[slice_new]

        # print("ROI shapes")
        # print(ROI_orig.shape)
        # print(ROI_new.shape)

        #masks of the original images
        mask_orig = np.divide(ROI_orig[:,:,3:].astype(np.float32),255)
        mask_new = np.divide(ROI_new[:,:,3:].astype(np.float32),255)

        #mask of the intersection
        intersection_mask = (mask_orig * mask_new).astype(np.uint8)
        intersection_area = np.sum(intersection_mask)
        if intersection_area == 0: intersection_area = 1

        #mask that has 2 on the intersection of the original images
        #and 1 elsewhere
        denom_mask = np.ones_like(intersection_mask) + intersection_mask
        #partmax = np.stack([part1 + part2, np.ones(part1.shape)], axis = 2)
        #partmax = np.max(partmax, axis = 2)

        if gain_correction:
            # print("Gain correction active!")
#           gainvals = np.zeros(3, dtype=np.float32)
            denom = np.maximum(ROI_new[:,:,:3], np.ones_like(ROI_new[:,:,:3]))

            gain_ratios = np.divide(ROI_orig[:,:,:3].astype(np.float32), denom) * intersection_mask

            gainvals = np.sum(gain_ratios, axis=0, keepdims=False)
            gainvals = np.sum(gainvals, axis=0, keepdims=False)
            gainvals = gainvals / intersection_area
            #gainvals = np.concatenate((gainvals, [1]))
            gainvals = gainvals.reshape(1,1,-1)

            # print(f"Gain vals: {gainvals}")

            ROI_new[:,:,:3] = (np.minimum(ROI_new[:,:,:3].astype(np.float32) * gainvals, 
                255 * np.ones_like(ROI_new[:,:,:3]))).astype(np.uint8)

            self._gain_vals += gainvals
            self._gain_measurements += 1


        alpha1 = np.divide(mask_orig, denom_mask)
        alpha2 = 1.0 - alpha1

        sq_sum = 0
        if compute_mse:
            sq_sum = np.sum(np.power((ROI_orig - ROI_new) * intersection_mask, 2))

        dst = ROI_orig * alpha1 + ROI_new * alpha2

        newalpha = np.stack([ROI_orig[:,:,3:], ROI_new[:,:,3:]], axis=2)
        newalpha = np.max(newalpha, axis = 2)
        dst[:,:,3:] = newalpha

        img_orig[slice_orig] = dst

        return (img_orig if not compute_mse else img_orig, sq_sum, intersection_area)
            



    '''
        Stitches the provided images.
        target_offsets      - offsets (top, bottom, left, right) to add to the first image
        num_images          - number of images to stitch. If 'images'
                                provides a len method, num_images can be set to None.
                                Alternatively, num_images may be used to limit
                                the number of the processed images.
        write_intermediate  - True to write images on every step of the algo
        output_filename     - output file name without an extension (possibly with path)
                                Set to None to skip writing to file
        gain_correction     - detects a required gain correction
                                by averaging the pre-channel ratio on the intersection
    '''
    def stitch(self, images : Iterable[np.ndarray], target_offsets, num_images = None, 
            write_intermediate = False, output_filename = "stitcher_output",
            gain_correction : GainCorrection = GainCorrection.LOCAL_AND_GLOBAL,
            compute_mse = False) -> np.ndarray:
        if num_images is None: num_images = len(images)
        if num_images == 0: return None
        if num_images == 1: return next(iter(images))

        self._gain_vals = np.zeros(3, dtype=np.float32).reshape(1,1,-1)
        self._gain_measurements = 0

        # target_width, target_height = target_size

        #homography accumalator: used to position a new patch wrt to the first image
        H_acc = np.eye(3) 
        target = None
        #keypoints and descriptors of the last image processed
        keypoints_acc = None 
        descriptors_acc = None

        mse = 0
        area = 0

        for i, img in tqdm(enumerate(images), "Stitching", total = num_images):
            if i >= num_images: break
            if target is None:
                #first image
                
                # time to get some sketchy sh*t, doo-daa, doo-daa
                # hope, i'll get away with it, doo-da-doo-da-da
                top, bottom, left, right = target_offsets
                # top = (target_height - img.shape[1]) // 2
                # bottom = target_height - img.shape[1] - top
                # left = (target_width - img.shape[0]) // 2
                # right = target_width - img.shape[0] - left

                target = cv2.copyMakeBorder(img, top, bottom, left, right,
                    cv2.BORDER_CONSTANT, value = [0,0,0,0])

                H_acc = np.array([[1, 0, left], [0, 1, top], [0, 0, 1]])

                # target = cv2.copyMakeBorder(img, 0, target_height - img.shape[1], 
                #     0, target_width - img.shape[0], cv2.BORDER_CONSTANT, 
                #     value = [0,0,0,0])


                keypoints_acc, descriptors_acc = self.detector.detectAndCompute(img, None)

            else:
                #successive image
                keypoints_new, descriptors_new = self.detector.detectAndCompute(img, None)

                homograpy = self.getHomography(keypoints_new, descriptors_new,
                        keypoints_acc, descriptors_acc)

                H_acc = np.matmul(H_acc, homograpy)

                #check where the homography sends the center of the new image
                center = np.divide(img.shape[:3], 2).astype(int)
                center[2] = 1 #to projective coordinates
                center_image = np.matmul(H_acc, center)
                shift = np.divide(center_image[:2], center_image[2]) - center[:2]
                shift = shift.astype(np.int32)
                # print(f"Shift: {shift}")

                sqrt2 = int(np.ceil(np.sqrt(2)).item())
                scale = (float(sqrt2) - 1) / 2

                #ROI size is picked so that any rotation would fit
                ROI_size = np.multiply(img.shape[:2], sqrt2)
                # print(f"ROI size: {ROI_size}")

                #new shift 
                desired_shift = np.multiply(img.shape[:2], scale).astype(np.int32)
                new_shift = desired_shift - shift
                new_shift_mat = np.array([[1, 0, new_shift[0]], [0, 1, new_shift[1]], [0, 0, 1]])
                new_homography = np.matmul(new_shift_mat, H_acc)

                #sanity check:
                # center_image_sanity = np.matmul(new_homography, center)
                # print(f"Center image: {center_image}")
                # print(f"Center image: {np.divide(center_image[:2], center_image[2])}")
                # shift_sanity = np.divide(center_image_sanity[:2], center_image_sanity[2]) -\
                        # center[:2]
                # shift_sanity = shift_sanity.astype(np.int32)
                # print(f"Shift sanity check: {shift_sanity}")

                warped_image = cv2.warpPerspective(img, new_homography, ROI_size, borderValue=[0,0,0,0])

                result = self.binAlphaBlend(target, warped_image, shift-desired_shift, 
                    gain_correction > 0, compute_mse)
                if compute_mse:
                    result, mse_val, mse_area = result
                    mse += mse_val
                    area += mse_area

                keypoints_acc, descriptors_acc = keypoints_new, descriptors_new

                if write_intermediate and output_filename:
                    cv2.imwrite(f'{output_filename}_{i+1}.png', target)

        #global gain correction:
        if gain_correction == 2 and self._gain_measurements:
            self._gain_vals /= self._gain_measurements
            print("Average gain correction: ", self._gain_vals)
            self._apply_gain(target, np.ones_like(self._gain_measurements) / self._gain_vals)

        if output_filename:
            # print(f'{output_filename}.png')
            cv2.imwrite(f'{output_filename}.png', target)

        if compute_mse:
            return target, (mse/area).item()
        else:
            return target