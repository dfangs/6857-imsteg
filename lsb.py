import numpy as np
import numpy.typing as npt
from numpy.random import Generator, PCG64
from typing import TypeVar, Tuple

T = TypeVar('T', bound=np.integer)
IntArray = npt.NDArray[T]

def embed(cover_mat: IntArray, data: bytes, key: bytes = None, num_lsb: int = 1) -> IntArray:
    flat_cover = cover_mat.flatten()

    # Perform random permutation with RNG seeded by the key
    if key:
        perm, inv_perm = _generate_permutation(key, len(flat_cover))
        flat_cover = flat_cover[perm]

    # Check hiding capacity
    if 8*len(data) > len(flat_cover) * num_lsb:
        raise ValueError('data exceeded the hiding capacity')

    # NOTE: Match the dtype here (otherwise there would be a weird casting bug)
    data_bitarray = _split_from_bytes(data, num_lsb).astype(flat_cover.dtype)

    flat_cover[:len(data_bitarray)] &= ~((np.ones_like(data_bitarray) << num_lsb) - 1)  # Remove `num_lsb` LSBs
    flat_cover[:len(data_bitarray)] |= data_bitarray  # Write data into LSBs

    # Invert permutation
    if key:
        flat_cover = flat_cover[inv_perm]

    stego_mat = np.reshape(flat_cover, cover_mat.shape)
    return stego_mat

def extract(stego_mat: IntArray, key: bytes = None, num_lsb: int = 1) -> bytes:
    flat_stego = stego_mat.flatten()

    if key:
        perm, _ = _generate_permutation(key, len(flat_stego))
        flat_stego = flat_stego[perm]

    return _merge_to_bytes(flat_stego, num_lsb)

def _generate_permutation(key: bytes, n: int) -> Tuple[IntArray, IntArray]:
    seed = int.from_bytes(key, byteorder='big')
    rng = Generator(PCG64(seed))

    perm = rng.permutation(np.arange(n))
    inv_perm = np.arange(n)
    inv_perm[perm] = np.arange(n)

    return (perm, inv_perm)

def _split_from_bytes(data: bytes, num_lsb: int) -> IntArray:
    bitstring = ''.join('{0:08b}'.format(byte) for byte in data)  # Split individual bits
    # Pad with zeros to the right so len(bitstring) is multiple of `num_lsb`
    if len(bitstring) % num_lsb != 0:
        final_length = (len(bitstring)//num_lsb + 1) * num_lsb
        bitstring = bitstring.ljust(final_length, '0')

    # Split into clusters of `num_lsb`
    bitarray = np.array([bitstring[i:i+num_lsb] for i in range(0, len(bitstring), num_lsb)])
    return bitarray

def _merge_to_bytes(bitarray: IntArray, num_lsb: int) -> bytes:
    # First, convert to 32-bit number to avoid `binary_repr()` adding minus sign
    bitstring = ''.join([np.binary_repr(val, 32)[-num_lsb:] for val in bitarray])
    return bytes([int(bitstring[i:i+8], 2) for i in range(0, len(bitstring), 8)])
