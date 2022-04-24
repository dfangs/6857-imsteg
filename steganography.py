import cv2
import numpy as np
import numpy.typing as npt
from enum import Enum, auto
from image import Image, ImageArray
from typing import Union


DctArray = npt.NDArray[np.int32]

class StegMode(Enum):
    LSB = auto()
    DCT = auto()

def hide(img: Image, data: bytes, mode: StegMode = StegMode.LSB) -> ImageArray:
    """TODO: docstring"""
    if mode == StegMode.LSB:
        return Image(_hide_lsb(img.array, data))
    elif mode == StegMode.DCT:
        return Image(_hide_dct(img.array, data))
    else:
        raise ValueError('given mode is not supported')

def recover(img: Image, mode: StegMode = StegMode.LSB) -> bytes:
    """TODO: docstring"""
    if mode == StegMode.LSB:
        return _recover_lsb(img.array)
    elif mode == StegMode.DCT:
        return _recover_dct(img.array)
    else:
        raise ValueError('given mode is not supported')

def _hide_lsb(img: Union[ImageArray, DctArray], data: bytes) -> ImageArray:
    flat_img = img.flatten()
    # if 8*len(data) > len(flat_img):  # TODO fix
    #     raise ValueError('len(data) exceeds the hiding capacity')

    # Remove LSB
    flat_img &= (~np.zeros_like(flat_img) ^ 1)

    data_bitstring = ''.join('{0:08b}'.format(byte) for byte in data)  # Separate individual bits
    data_bitarray = np.array(list(data_bitstring), dtype=np.uint8)
    flat_img[:len(data_bitarray)] |= data_bitarray

    stego_img = np.reshape(flat_img, img.shape)
    return stego_img

def _recover_lsb(img: Union[ImageArray, DctArray]) -> bytes:
    flat_img = img.flatten()
    flat_img &= np.ones_like(flat_img)
    data = bytes([int(''.join(bitarray.astype(str)), 2) for bitarray in flat_img.reshape(-1, 8)])
    return data

def _hide_dct(img: ImageArray, data: bytes) -> ImageArray:
    dct_img = _dct(img)
    stego_dct_img = _hide_lsb(dct_img, data)
    return _idct(stego_dct_img)

def _recover_dct(img: ImageArray) -> bytes:
    dct_img = _dct(img)
    return _recover_lsb(dct_img)

def _dct(img: ImageArray) -> DctArray:
    if len(img.shape) != 2:
        raise ValueError('Image is not grayscale')

    dct_float = cv2.dct(img.astype(np.float32))
    return np.rint(dct_float).astype(np.int32)

def _idct(img: DctArray) -> ImageArray:
    if len(img.shape) != 2:
        raise ValueError('Image is not grayscale')

    idct_float = cv2.idct(img.astype(np.float32))
    return np.rint(idct_float).astype(np.uint8)
