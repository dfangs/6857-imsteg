import cv2
import math
import numpy as np
import numpy.typing as npt
from image import Image, ImageArray
from typing import Tuple, Union

DctArray = npt.NDArray[np.int32]

def _hide_pvd(cover_img: Union[ImageArray, DctArray], data: bytes, key: int) -> ImageArray:
    # let key = threshold
    # cover_img is not greyscale

    # TODO Fix issue with ending of message
    data += b'EOM:' 
    if len(cover_img.shape) != 3:
        raise ValueError("cover_img must be an array of rgb")
    
    threshold = key
    secret_bits = _bytes_to_bitarray(data)
    start = 0

    stego_img = np.zeros(cover_img.shape)

    for r in range(cover_img.shape[0]):
        for c in range(cover_img.shape[1]):
            rg_pair = cover_img[r, c, :2]
            gb_pair = cover_img[r, c, 1:]

            t1 = abs(float(rg_pair[0]) - float(rg_pair[1]))
            t2 = abs(float(gb_pair[0]) - float(gb_pair[1]))

            if t1 + t2 < threshold and start < len(secret_bits):
                rg_pair_new, start = _pvd_encrypt(rg_pair, secret_bits, start)
                gb_pair_new = gb_pair
                if start < len(secret_bits):
                    gb_pair_new, start = _pvd_encrypt(gb_pair, secret_bits, start)

                g_new = round((rg_pair_new[1] + gb_pair_new[0])/2.0)
                r_new = rg_pair_new[0] - (rg_pair_new[1] - g_new)
                b_new = gb_pair_new[1] - (gb_pair_new[0] - g_new)
                stego_img[r, c, :] = np.array([r_new, g_new, b_new])

            else:
                stego_img[r, c, :] = cover_img[r, c, :]
    
    return stego_img

def _recover_pvd(stego_img: Union[ImageArray, DctArray], key: int) -> bytes:
    if len(stego_img.shape) != 3:
        raise ValueError("stego_img must be an array of rgb")
    
    threshold = key
    secret_bits = np.array([], dtype = np.uint8)
    for r in range(stego_img.shape[0]):
        for c in range(stego_img.shape[1]):
            rgb_stego = stego_img[r, c, :]

            t1 = abs(float(rgb_stego[0]) - float(rgb_stego[1]))
            t2 = abs(float(rgb_stego[1]) - float(rgb_stego[2]))

            if t1 + t2 < threshold:
                extract_secret_bits = np.concatenate((_pvd_decrypt(rgb_stego[:2]), _pvd_decrypt(rgb_stego[1:])))
                secret_bits = np.concatenate((secret_bits, extract_secret_bits))

    secret_bytes = _bitarray_to_bytes(secret_bits)
    end_idx = secret_bytes.rindex(b'EOM')
    

    return secret_bytes[:end_idx]

### HELPER FUNCTIONS

def _pvd_get_range(d_old: np.uint8) -> Tuple:
    # get the quantize range for given block
    quantize_ranges = [(0, 7), (8, 15), (16, 31), (32, 63), (64, 127), (128, 255)]

    lower, upper = -1, -1
    for quantize_lower, quantize_upper in quantize_ranges:
        if d_old >= quantize_lower and d_old <= quantize_upper:
            lower, upper = quantize_lower, quantize_upper
            break
    return lower, upper

def _pvd_decrypt(block: cv2.Mat) -> npt.NDArray[np.uint8]:
    # TODO: this almost works except near the end where the bits don't fill capacity
    d_new = int(abs(float(block[0]) - float(block[1])))
    lower, upper = _pvd_get_range(d_new)
    t = math.ceil(math.log2(upper - lower + 1)) # number of bits can store
    extract_secret = [int(char) for char in bin(d_new - lower)[2:]]
    
    return np.array( [0] * (t - len(extract_secret)) + extract_secret , dtype = np.uint8)

def _pvd_encrypt(block: cv2.Mat, secret_bits: npt.NDArray[np.uint8], start: int) -> Tuple:
    """
    :param block: is 1d 
    :param secret_bits: the entire secret bit
    :param start: where to start in the secret
    :returns a tuple of the new pixel pair (cv2.Mat), next index to start secret_bits at

    An RGB colour imagesteganography scheme using overlapping block-based pixel-value differencing
    """
    d_old = abs(float(block[0]) - float(block[1]))
    lower, upper = _pvd_get_range(d_old)
    t = math.ceil(math.log2(upper - lower + 1)) # number of bits can store

    bit_array = np.zeros((8,), dtype = np.uint8)
    num_bits_left = min(t, len(secret_bits) - start)

    # padding in case there are not enough bits left!
    
    bit_array[-num_bits_left:] = secret_bits[start:start+num_bits_left]
    secret_to_hide = np.packbits(bit_array)[0]

    d_new = lower + secret_to_hide
    m = d_new - d_old

    return _pvd_get_stego_pixels(block, m), start+t


def _pvd_get_stego_pixels(block: cv2.Mat, m: int) -> cv2.Mat:
    """
    :param block: is 1d 
    :param m: difference of d' and d (NO ABS)
    """
    m_abs = abs(m)
    if block[0] >= block[1] and m > 0:
        return np.array([block[0] + math.ceil(m_abs/2.0), block[1] - math.floor(m_abs/2.0)])
    elif block[0] < block[1] and m > 0:
        return np.array([block[0] - math.floor(m_abs/2.0), block[1] + math.ceil(m_abs/2.0)])
    elif block[0] >= block[1] and m <= 0:
        return np.array([block[0] - math.ceil(m_abs/2.0), block[1] + math.floor(m_abs/2.0)])
    else:
        return np.array([block[0] + math.ceil(m_abs/2.0), block[1] - math.floor(m_abs/2.0)])
    

def _bitarray_to_bytes(bitarray: npt.NDArray[np.uint8]) -> bytes:
    bitstring = ''.join(bitarray.astype(str))
    return bytes([int(bitstring[8*i : 8*(i+1)], 2) for i in range(len(bitstring) // 8)])


def _bytes_to_bitarray(data: bytes) -> npt.NDArray[np.uint8]:
    bitstring = ''.join('{0:08b}'.format(byte) for byte in data)  # Separate individual bits
    bitarray = np.array(list(bitstring), dtype=np.uint8)
    return bitarray

if __name__ == "__main__":

    cover_img = Image.from_file('images/testimage1_128x96.jpg').array
    secret_text = None
    with open('secret.txt', 'rb') as f:
        secret_text = f.read()
    
    secret_text = secret_text[:100]
    encrypt = _hide_pvd(cover_img, secret_text, 1000)
    cv2.imwrite("pvd.png", encrypt)
    decrypt = _recover_pvd(encrypt, 1000)
    print(decrypt)