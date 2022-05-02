import numpy as np
import numpy.typing as npt
from typing import TypeVar, List

T = TypeVar('T', bound=np.integer)
BitArray = npt.NDArray[np.uint8]
IntArray = npt.NDArray[T]

def bytes_to_bitarray(data: bytes) -> BitArray:
    bitstring = ''.join('{0:08b}'.format(byte) for byte in data)  # Separate individual bits
    bitarray = np.array(list(bitstring), dtype=np.uint8)
    return bitarray

def bitarray_to_bytes(bitarray: BitArray) -> bytes:
    bitstring = ''.join(bitarray.astype(str))
    return bytes([int(bitstring[8*i : 8*(i+1)], 2) for i in range(len(bitstring) // 8)])

def separate_channels(array: IntArray) -> List[IntArray]:
    assert len(array.shape) == 3
    return [array[..., i] for i in range(array.shape[2])]

def combine_channels(channels: List[IntArray]) -> IntArray:
    return np.stack(channels, axis=2)
