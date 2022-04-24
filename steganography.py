import cv2
import numpy as np
import numpy.typing as npt
from enum import Enum, auto
from image import Image, ImageArray
from typing import Union


DctArray = npt.NDArray[np.int32]

class Mode(Enum):
    """Represents various mode of steganography supported by this package."""
    LSB = auto()
    DCT_LSB = auto()
    DCT_ADD = auto()


### PUBLIC API

def hide(cover_img: Image, data: bytes, mode: Mode, key: bytes = None) -> Image:
    """Conceals a sequence of data within the given cover image using steganography.

    Args:
        cover_img: the medium image in which the data will be concealed.
        data: sequence of data to conceal.
        mode: mode of steganography used.
        key: an optional symmetric-key used to perform steganography.
    Returns:
        An `Image` representing the stego-image of the concealed data.
    Raises:
        ValueError: if the given combination of mode and key is not supported.

    """
    if mode == Mode.LSB:
        return Image(_hide_lsb(cover_img.array, data, key))
    elif mode == Mode.DCT_LSB:
        return Image(_hide_dct_lsb(cover_img.array, data, key))
    elif mode == Mode.DCT_ADD and key:
        return Image(_hide_dct_add(cover_img.array, data, key))
    else:
        raise ValueError('The given combination of mode and key is not supported')

def recover(stego_img: Image, mode: Mode, key: bytes = None) -> bytes:
    """Recovers concealed data from the given stego-image.

    Args:
        stego_img: the image from which the data will be recovered.
        mode: mode of steganography used.
        key: an optional symmetric-key used to perform steganography.
    Returns:
        A sequence of bytes recovered from the stego-image.
    Raises:
        ValueError: if the given combination of mode and key is not supported.
    """
    if mode == Mode.LSB:
        return _recover_lsb(stego_img.array, key)
    elif mode == Mode.DCT_LSB:
        return _recover_dct_lsb(stego_img.array, key)
    elif mode == Mode.DCT_ADD and key:
        return _recover_dct_add(stego_img.array, key)
    else:
        raise ValueError('The given combination of mode and key is not supported')


### IMPLEMENTATION FUNCTIONS

def _hide_lsb(cover_img: Union[ImageArray, DctArray], data: bytes, key: bytes) -> ImageArray:
    flat_img = cover_img.flatten()
    flat_img &= (~np.zeros_like(flat_img) ^ 1)  # Remove LSB

    if key:
        pass
    else:
        # TODO: Check hiding capacity
        # if 8*len(data) > len(flat_img):
        #     raise ValueError('len(data) exceeds the hiding capacity')

        data_bitarray = _bytes_to_bitarray(data)
        flat_img[:len(data_bitarray)] |= data_bitarray

    stego_img = np.reshape(flat_img, cover_img.shape)
    return stego_img

def _recover_lsb(stego_img: Union[ImageArray, DctArray], key: bytes) -> bytes:
    flat_img = stego_img.flatten()
    flat_img &= np.ones_like(flat_img)

    if key:
        pass
    else:
        data = _bitarray_to_bytes(flat_img)

    return data

def _hide_dct_lsb(cover_img: ImageArray, data: bytes, key: bytes) -> ImageArray:
    dct_img = _dct(cover_img)
    stego_dct_img = _hide_lsb(dct_img, data, key)
    return _idct(stego_dct_img)

def _recover_dct_lsb(stego_img: ImageArray, key: bytes) -> bytes:
    dct_img = _dct(stego_img)
    return _recover_lsb(dct_img, key)

def _hide_dct_add(cover_img: ImageArray, data: bytes, key: bytes) -> ImageArray:
    assert key

    dct_img = _dct(cover_img)
    flat_dct_img = dct_img.flatten()

    # TODO: Fix
    data_bitarray = _bytes_to_bitarray(data)
    flat_dct_img[:len(data_bitarray)] += data_bitarray

    stego_img = np.reshape(flat_dct_img, cover_img.shape)
    return _idct(stego_img)

def _recover_dct_add(stego_img: ImageArray, key: bytes) -> bytes:
    assert key

    # TODO: Fix
    dct_img = _dct(stego_img)
    return _recover_lsb(dct_img)


### HELPER FUNCTIONS

def _bytes_to_bitarray(data: bytes) -> npt.NDArray[np.uint8]:
    bitstring = ''.join('{0:08b}'.format(byte) for byte in data)  # Separate individual bits
    bitarray = np.array(list(bitstring), dtype=np.uint8)
    return bitarray

def _bitarray_to_bytes(bitarray: npt.NDArray[np.uint8]) -> bytes:
    bitstring = ''.join(bitarray.astype(str))
    return bytes([int(bitstring[8*i : 8*(i+1)], 2) for i in range(len(bitstring) // 8)])

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
