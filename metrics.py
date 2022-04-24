import skimage.metrics as skmetrics
import numpy as np
import cv2
from enum import Enum

import rs_steganalysis as rsstegan

class Metric(Enum):
    PSNR = 1
    RM = 2
    SM = 3
    RM_NEG = 4
    SM_NEG = 5

def get_psnr(cover_img: cv2.Mat, stego_img: cv2.Mat) -> float:
    """
    :param cover_img: the original image to hide secret
    :param stego_img: the image containing the secret
    :return: the psnr value for stego_img given original cover_img
    If it is in between 30 dB and 40 dB, can be acceptable, but a PSNR less than 30 dB is not acceptable because the distortion is very high.
    (ref: Performance Evaluation Parameters of Image Steganography Techniques)
    """

    return skmetrics.peak_signal_noise_ratio(cover_img, stego_img)

def get_rs(stego_img: cv2.Mat) -> tuple:
    """
    :param img: input image
    :return: tuple of percentage of (Rm, Sm, R-m, S-m) groups
    """

    # This mask is used in the original RS steganography attack paper
    mask = np.array([[0, 1, 1, 0]])

    rm, sm = rsstegan.calculate_count_groups(stego_img, mask)
    rm_neg, sm_neg = rsstegan.calculate_count_groups(stego_img, -mask)

    return (rm, sm, rm_neg, sm_neg)

if __name__ == "__main__":
    print("hello")




