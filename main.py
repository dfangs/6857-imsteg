from image import Image
import steganography as stego


if __name__ == '__main__':
    cover_img = Image.from_file('images/mit_512x384.jpg')
    hidden_img = Image.from_file('images/mit_64x48.jpg')
    message = b'I hate crypto'

    # (1) Hide text message inside cover image using LSB
    stego_img = stego.hide(cover_img, message, mode=stego.Mode.LSB)
    recovered_data = stego.recover(stego_img, mode=stego.Mode.LSB)
    print(recovered_data.decode('utf-8'))

    # (2) Hide an (unencrypted) image inside cover image using LSB
    stego_img = stego.hide(cover_img, hidden_img.to_bytes(), mode=stego.Mode.LSB)
    recovered_data = stego.recover(stego_img, mode=stego.Mode.LSB)
    truncated_recovered_data = recovered_data[:len(hidden_img.to_bytes())]  # TODO: There shouldn't be the need to truncate?
    assert hidden_img.to_bytes() == truncated_recovered_data

    recovered_img = Image.from_bytes(truncated_recovered_data, hidden_img.shape)
    # recovered_img.show()

    # (3) Hide an (unencrypted) image inside cover image using DCT
    stego_img = stego.hide(cover_img, hidden_img.to_bytes(), mode=stego.Mode.DCT_LSB)
    # stego_img.show()
    cover_img.to_grayscale().save('output/mit_grayscale.jpg')
    stego_img.save('output/mit_stego_img.jpg')

    recovered_data = stego.recover(stego_img, mode=stego.Mode.DCT_LSB)
    truncated_recovered_data = recovered_data[:len(hidden_img.to_bytes())]
    recovered_img = Image.from_bytes(truncated_recovered_data, hidden_img.shape)
    # recovered_img.show()
    recovered_img.save('output/mit_recovered_dct.jpg')
