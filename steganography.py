import numpy as np
import numpy.typing as npt
import util
import lsb
from dct import dct, idct
from enum import Enum, auto
from image import Image, ImageArray
from typing import Union

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
        return _hide_lsb(cover_img, data, key)
    elif mode == Mode.DCT_LSB:
        return _hide_dct_lsb(cover_img, data, key)
    elif mode == Mode.DCT_ADD and key:
        return _hide_dct_add(cover_img, data, key)
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
        return _recover_lsb(stego_img, key)
    elif mode == Mode.DCT_LSB:
        return _recover_dct_lsb(stego_img, key)
    elif mode == Mode.DCT_ADD and key:
        return _recover_dct_add(stego_img, key)
    else:
        raise ValueError('The given combination of mode and key is not supported')


### IMPLEMENTATION FUNCTIONS

def _hide_lsb(cover_img: Image, data: bytes, key: bytes) -> Image:
    return Image(lsb.embed(cover_img.array, data, key))

def _recover_lsb(stego_img: Image, key: bytes) -> bytes:
    return lsb.extract(stego_img.array, key)

def _hide_dct_lsb(cover_img: Image, data: bytes, key: bytes) -> Image:
    if len(cover_img.shape) == 3:
        ycrcb_img = cover_img.to_ycrcb()

        dct_array = util.combine_channels(
            [dct(channel) for channel in util.separate_channels(ycrcb_img.array)]
        )
        stego_dct_array = lsb.embed(dct_array, data, key)

        return Image(util.combine_channels(
            [idct(channel) for channel in util.separate_channels(stego_dct_array)]
        )).to_rgb()

    else:
        dct_array = dct(cover_img.array)
        stego_dct_array = lsb.embed(dct_array, data, key)
        return Image(idct(stego_dct_array)).to_rgb()

def _recover_dct_lsb(stego_img: Image, key: bytes) -> bytes:
    if len(stego_img.shape) == 3:
        ycrcb_img = stego_img.to_ycrcb()
        dct_array = util.combine_channels(
            [dct(channel) for channel in util.separate_channels(ycrcb_img.array)]
        )
    else:
        dct_array = dct(stego_img.array)

    return lsb.extract(dct_array, key)

def _hide_dct_add(cover_img: Image, data: bytes, key: bytes) -> Image:
    assert key

    dct_img = dct(cover_img)
    flat_dct_img = dct_img.flatten()

    # TODO: Fix
    data_bitarray = util.bytes_to_bitarray(data)
    flat_dct_img[:len(data_bitarray)] += data_bitarray

    stego_img = np.reshape(flat_dct_img, cover_img.shape)
    return idct(stego_img)

def _recover_dct_add(stego_img: Image, key: bytes) -> bytes:
    assert key

    # TODO: Fix
    dct_img = dct(stego_img)
    return _recover_lsb(dct_img)
