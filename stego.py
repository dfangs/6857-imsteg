import cv2
import numpy as np

def hide_lsb(img: cv2.Mat, data: bytes) -> cv2.Mat:
    flat_img = img.flatten()
    if len(data) > len(flat_img):
        raise ValueError('len(data) exceeds the hiding capacity')

    # Remove LSB
    flat_img &= (~np.zeros_like(flat_img) - 1)

    data_bitstring = ''.join('{0:08b}'.format(byte) for byte in data)
    data_bitarray = np.array([int(bit) for bit in data_bitstring], dtype='uint8')
    flat_img[:len(data_bitarray)] |= data_bitarray

    stego_img = np.reshape(flat_img, img.shape)
    return stego_img

def recover_lsb(img: cv2.Mat) -> bytes:
    flat_img = img.flatten()
    flat_img &= np.ones_like(flat_img)
    data = bytes([int(''.join(bitarray.astype(str)), 2) for bitarray in flat_img.reshape(-1, 8)])

    return data

if __name__ == '__main__':
    filepath = 'cover_image.jpg'
    cover_img = cv2.imread(filepath)
    hidden_text = 'I hate crypto'

    stego_img = hide_lsb(cover_img, bytes(hidden_text, 'utf-8'))
    # cv2.imshow('img', stego_img)
    cv2.waitKey(0)

    # print((cover_img - stego_img).flatten()[:100])
    recovered_text = recover_lsb(stego_img).decode('utf-8')
    print(recovered_text)

    # gray_img = cv2.cvtColor(cover_img, cv2.COLOR_BGR2GRAY)
    # dctr = cv2.dct(np.float32(gray_img))
    # idct = cv2.dct(dctr, cv2.DCT_INVERSE)
    # print(idct)
    # print(gray_img)  # TODO: doesn't match for now
