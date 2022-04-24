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

def _hide_lsb(img: Union[ImageArray, DctArray], data: bytes, bitlen: int = 8) -> ImageArray:
    flat_img = img.flatten()
    if 8*len(data) > len(flat_img):  # TODO
        raise ValueError('len(data) exceeds the hiding capacity')

    # Remove LSB
    flat_img &= (~np.zeros_like(flat_img) - 1)

    data_bitstring = ''.join(f'{{0:0{bitlen}b}}'.format(byte) for byte in data)
    data_bitarray = np.array([int(bit) for bit in data_bitstring], dtype='uint8')
    flat_img[:len(data_bitarray)] |= data_bitarray

    stego_img = np.reshape(flat_img, img.shape)
    return stego_img

def _recover_lsb(img: ImageArray) -> bytes:
    flat_img = img.flatten()
    flat_img &= np.ones_like(flat_img)
    data = bytes([int(''.join(bitarray.astype(str)), 2) for bitarray in flat_img.reshape(-1, 8)])

    return data

def _hide_dct(img: ImageArray, data: bytes) -> ImageArray:
    dct_img = _dct(img)
    stego_dct_img = _hide_lsb(dct_img, data, bitlen=32)
    idct_img = _idct(stego_dct_img)
    return idct_img

def _recover_dct(img: ImageArray) -> bytes:
    dct_img = _idct(img)
    return _recover_lsb(dct_img)

def _dct(img: ImageArray) -> DctArray:
    if len(img.shape) != 2:
        raise ValueError('Image is not grayscale')

    dct = cv2.dct(img.astype(np.float32))
    rounded_dct = np.rint(dct)

    return rounded_dct.astype(np.int32)

def _idct(dct_img: DctArray) -> ImageArray:
    if len(dct_img.shape) != 2:
        raise ValueError('Image is not grayscale')

    idct = cv2.idct(dct_img.astype(np.float32))
    rounded_idct = np.rint(idct)

    return rounded_idct.astype(np.uint8)
