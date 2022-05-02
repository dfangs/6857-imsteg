from __future__ import annotations
import cv2
import numpy as np
import numpy.typing as npt


ImageMatrix = npt.NDArray[np.uint8]

class Image:
    def __init__(self, matrix: ImageMatrix) -> None:
        self.matrix = matrix

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
        return bytes(self.matrix.flatten())

    def to_grayscale(self) -> Image:
        return Image(cv2.cvtColor(self.matrix, cv2.COLOR_BGR2GRAY))

    @staticmethod
    def ycrcb_to_rgb(matrix: ImageMatrix) -> ImageMatrix:
        return cv2.cvtColor(matrix, cv2.COLOR_YCrCb2BGR)

    @staticmethod
    def rgb_to_ycrcb(matrix: ImageMatrix) -> ImageMatrix:
        return cv2.cvtColor(matrix, cv2.COLOR_BGR2YCrCb)


    @property
    def shape(self):
        return self.matrix.shape

    def show(self) -> None:
        cv2.imshow('img', self.matrix)
        cv2.waitKey(0)

    def save(self, filename: str) -> None:
        cv2.imwrite(filename, self.matrix)
