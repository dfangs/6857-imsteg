import numpy as np
import util
from util import IntArray

def embed(img_array: IntArray, data: bytes, key: bytes) -> IntArray:
    flat_img = img_array.flatten()
    flat_img &= (~np.zeros_like(flat_img) ^ 1)  # Remove LSB

    if key:
        pass
    else:
        # TODO: Check hiding capacity
        # if 8*len(data) > len(flat_img):
        #     raise ValueError('len(data) exceeds the hiding capacity')

        data_bitarray = util.bytes_to_bitarray(data)
        flat_img[:len(data_bitarray)] |= data_bitarray

    stego_img = np.reshape(flat_img, img_array.shape)
    return stego_img

def extract(img_array: IntArray, key: bytes) -> bytes:
    flat_img = img_array.flatten()
    flat_img &= np.ones_like(flat_img)

    if key:
        pass
    else:
        data = util.bitarray_to_bytes(flat_img)

    return data
