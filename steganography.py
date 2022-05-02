import lsb
from dct import dct, idct
from enum import Enum, auto
from image import Image

class Mode(Enum):
    """Represents various mode of steganography supported by this package."""
    LSB = auto()
    DCT = auto()
    JSTEG = auto()

### PUBLIC API

def hide(cover_img: Image, data: bytes, mode: Mode, **kwargs) -> Image:
    """Conceals a sequence of data within the given cover image using steganography.

    Args:
        cover_img: the medium image in which the data will be concealed.
        data: sequence of data to conceal.
        mode: mode of steganography used.
        key: an optional symmetric-key used to perform steganography.  # TODO: Remove
    Returns:
        An `Image` representing the stego-image of the concealed data.
    Raises:
        ValueError: if the given combination of mode and key is not supported.

    """
    if mode == Mode.LSB:
        return _hide_lsb(cover_img, data, **kwargs)
    elif mode == Mode.DCT:
        return _hide_dct(cover_img, data, **kwargs)
    else:
        raise ValueError('The given combination of mode and key is not supported')

def recover(stego_img: Image, mode: Mode, **kwargs) -> bytes:
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
        return _recover_lsb(stego_img, **kwargs)
    elif mode == Mode.DCT:
        return _recover_dct(stego_img, **kwargs)
    else:
        raise ValueError('The given combination of mode and key is not supported')


### IMPLEMENTATION FUNCTIONS

def _hide_lsb(cover_img: Image, data: bytes, key: bytes = None, num_lsb: int = 1) -> Image:
    return Image(lsb.embed(cover_img.matrix, data, key, num_lsb))

def _recover_lsb(stego_img: Image, key: bytes = None, num_lsb: int = 1) -> bytes:
    return lsb.extract(stego_img.matrix, key, num_lsb)

def _hide_dct(cover_img: Image, data: bytes, key: bytes = None, num_lsb: int = 1) -> Image:
    # YCrCb allows for more efficient compression
    cover_mat = cover_img.matrix
    if len(cover_mat.shape) == 3:
        cover_mat = Image.rgb_to_ycrcb(cover_mat)

    dct_mat = dct(cover_mat)
    stego_dct_mat = lsb.embed(dct_mat, data, key, num_lsb)
    stego_mat = idct(stego_dct_mat)

    if len(cover_mat.shape) == 3:
        stego_mat = Image.ycrcb_to_rgb(stego_mat)

    return Image(stego_mat)

def _recover_dct(stego_img: Image, key: bytes = None, num_lsb: int = 1) -> bytes:
    stego_mat = stego_img.matrix
    if len(stego_mat.shape) == 3:
        stego_mat = Image.rgb_to_ycrcb(stego_mat)

    return lsb.extract(dct(stego_mat), key, num_lsb)
