from image import ImageArray
import cv2
import numpy as np
import numpy.typing as npt

DctInputArray = npt.NDArray[np.int16]   # Original image is in uint8, but we extend it to int16 for rescaling
DctOutputArray = npt.NDArray[np.int32]  # Must be 32-bit since DCT produces 32-bit float

def dct(img: ImageArray) -> DctOutputArray:
    if len(img.shape) != 2:
        raise ValueError('Image is not grayscale')
    nrows, ncols = img.shape

    signed_img = img.astype(np.int16) - 128  # Recenter [0, 255] -> [-128, 127]
    dct_img = np.zeros_like(img, dtype=np.int32)

    # Perform DCT block-wise, where a block is 8x8 pixels
    for r in range(nrows//8):
        for c in range(ncols//8):
            block = signed_img[8*r:8*(r+1), 8*c:8*(c+1)]
            dct_img[8*r:8*(r+1), 8*c:8*(c+1)] = _dct(block)

    return dct_img

def idct(img: DctOutputArray) -> ImageArray:
    if len(img.shape) != 2:
        raise ValueError('Image is not grayscale')
    nrows, ncols = img.shape

    idct_img = np.zeros_like(img, dtype=np.int8)

    # Perform IDCT block-wise, where a block is 8x8 pixels
    for r in range(nrows//8):
        for c in range(ncols//8):
            block = img[8*r:8*(r+1), 8*c:8*(c+1)]
            idct_img[8*r:8*(r+1), 8*c:8*(c+1)] = _idct(block)

    return (idct_img + 128).astype(np.uint8)  # Recenter [0, 255] -> [-128, 127]

def _dct(img: DctInputArray) -> DctOutputArray:
    dct_float = cv2.dct(img.astype(np.float32))
    return dct_float.astype(np.int32)

def _idct(img: DctOutputArray) -> DctInputArray:
    idct_float = cv2.idct(img.astype(np.float32))
    return idct_float.astype(np.int16)
