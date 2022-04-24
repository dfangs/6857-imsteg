from image import Image
import cv2
import steganography as stg
import numpy as np


if __name__ == '__main__':
    cover_img = Image.from_file('images/testimage1_512x384.jpg')
    hidden_img = Image.from_file('images/testimage1_96x72.jpg')
    message = b'I hate crypto'

    # (1) Hide text message inside cover image using LSB
    stego_img = stg.hide(cover_img, message, mode=stg.StegMode.LSB)
    recovered_data = stg.recover(stego_img, mode=stg.StegMode.LSB)
    print(recovered_data.decode('utf-8'))

    # (2) Hide an (unencrypted) image inside cover image using LSB
    stego_img = stg.hide(cover_img, hidden_img.to_bytes(), mode=stg.StegMode.LSB)
    recovered_data = stg.recover(stego_img, mode=stg.StegMode.LSB)
    truncated_recovered_data = recovered_data[:len(hidden_img.to_bytes())]  # TODO: There shouldn't be the need to truncate?
    assert hidden_img.to_bytes() == truncated_recovered_data

    recovered_img = Image.from_bytes(truncated_recovered_data, hidden_img.shape)
    # recovered_img.show()

    # (3) Hide an (unencrypted) image inside cover image using DCT
    stego_img = stg.hide(cover_img.to_grayscale(), hidden_img.to_bytes(), mode=stg.StegMode.DCT)
    # stego_img.show()
    recovered_data = stg.recover(stego_img, mode=stg.StegMode.DCT)
    truncated_recovered_data = recovered_data[:len(hidden_img.to_bytes())]
    # assert hidden_img.to_bytes() == truncated_recovered_data
    recovered_img = Image.from_bytes(truncated_recovered_data, hidden_img.shape)
    recovered_img.show()
