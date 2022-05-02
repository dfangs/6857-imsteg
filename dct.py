from image import ImageMatrix
import cv2
import numpy as np
import numpy.typing as npt

DctInputMatrix = npt.NDArray[np.int16]   # Original image is in uint8, but we extend it to int16 for rescaling
DctOutputMatrix = npt.NDArray[np.int32]  # Must be 32-bit since DCT produces 32-bit float

BLOCK_SIZE = 8

def dct(mat: ImageMatrix) -> DctOutputMatrix:
    nrows, ncols = mat.shape[0], mat.shape[1]
    nchannels = 1 if len(mat.shape) == 2 else mat.shape[2]

    if nrows % BLOCK_SIZE != 0 or ncols % BLOCK_SIZE != 0:
        raise ValueError(f'Width and height of the image must be multiple of {BLOCK_SIZE}')

    # Temporarily expand grayscale matrix into 1-channel image
    if len(mat.shape) == 2:
        mat = np.expand_dims(mat, axis=2)

    # Recenter [0, 255] -> [-128, 127]
    signed_mat = mat.astype(np.int16) - 128

    # Perform DCT block-wise
    dct_mat = np.zeros_like(signed_mat, dtype=np.int32)
    for i in range(0, nrows, BLOCK_SIZE):
        for j in range(0, ncols, BLOCK_SIZE):
            for k in range(nchannels):
                block = signed_mat[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE, k]
                dct_mat[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE, k] = _dct(block)

    # Revert matrix expansion
    if len(dct_mat[0][0]) == 1:
        dct_mat = np.squeeze(dct_mat, axis=2)

    return dct_mat

def idct(dct_mat: DctOutputMatrix) -> ImageMatrix:
    nrows, ncols = dct_mat.shape[0], dct_mat.shape[1]
    nchannels = 1 if len(dct_mat.shape) == 2 else dct_mat.shape[2]

    if nrows % BLOCK_SIZE != 0 or ncols % BLOCK_SIZE != 0:
        raise ValueError(f'Width and height of the image must be multiple of {BLOCK_SIZE}')

    # Temporarily expand grayscale matrix into 1-channel image
    if len(dct_mat.shape) == 2:
        dct_mat = np.expand_dims(dct_mat, axis=2)

    # Perform IDCT block-wise
    idct_mat = np.zeros_like(dct_mat, dtype=np.int16)
    for i in range(0, nrows, BLOCK_SIZE):
        for j in range(0, ncols, BLOCK_SIZE):
            for k in range(nchannels):
                block = dct_mat[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE, k]
                idct_mat[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE, k] = _idct(block)

    # Revert matrix expansion
    if len(idct_mat[0][0]) == 1:
        idct_mat = np.squeeze(idct_mat, axis=2)

    # Floor and ceil out-of-bound values
    idct_mat = np.minimum(np.maximum(idct_mat, -128), 127)

    return (idct_mat + 128).astype(np.uint8)  # Recenter [-128, 127] -> [0, 255]

def _dct(img: DctInputMatrix) -> DctOutputMatrix:
    dct_float = cv2.dct(img.astype(np.float32))
    return np.rint(dct_float).astype(np.int32)  # Round to nearest int

def _idct(img: DctOutputMatrix) -> DctInputMatrix:
    idct_float = cv2.idct(img.astype(np.float32))
    return np.rint(idct_float).astype(np.int16)  # Round to nearest int
