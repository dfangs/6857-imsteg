from __future__ import annotations
import cv2
import numpy as np
import numpy.typing as npt


ImageArray = npt.NDArray[np.uint8]

class Image:
    def __init__(self, array: ImageArray) -> None:
        self.array = array

    @classmethod
    def from_file(cls, filename: str, grayscale: bool =False) -> Image:
        if grayscale:
            return cls(cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2GRAY))
        return cls(cv2.imread(filename))

    @classmethod
    def from_bytes(cls, raw_bytes: bytes, shape) -> Image:
        array = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(shape)
        return cls(array)

    def to_bytes(self) -> bytes:
        return bytes(self.array.flatten())

    # TODO: This might not be needed eventually
    def to_grayscale(self) -> Image:
        print(Image(cv2.cvtColor(self.array, cv2.COLOR_BGR2GRAY)).array.shape)
        return Image(cv2.cvtColor(self.array, cv2.COLOR_BGR2GRAY))

    @property
    def shape(self):
        return self.array.shape

    def show(self) -> None:
        cv2.imshow('img', self.array)
        cv2.waitKey(0)
