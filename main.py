from image import Image
from metrics import get_psnr
import steganography as stego

def run_stego_all(img_name: str,
        cover_img: Image,
        hidden_img: Image,
        secret_key: bytes,
    ):
    for mode in stego.Mode:
        for key in (None, secret_key):
            for num_lsb in (1, 2, 3):
                run_stego_once(img_name, cover_img, hidden_img, mode, key, num_lsb)

def run_stego_once(
        img_name: str,
        cover_img: Image,
        hidden_img: Image,
        mode: stego.Mode,
        key: bytes = None,
        num_lsb: int = 1,
    ):
    with_key = '_secure' if key else ''

    # Hide image
    stego_img = stego.hide(cover_img, hidden_img.to_bytes(), mode=mode, num_lsb=num_lsb, key=key)
    stego_img.save(f'output/{img_name}_stego_{mode.name}_{num_lsb}bit{with_key}.jpg')
    print(get_psnr(cover_img.matrix, stego_img.matrix))

    # Recover image
    recovered_data = stego.recover(stego_img, mode=mode, num_lsb=num_lsb, key=key)
    truncated_recovered_data = recovered_data[:len(hidden_img.to_bytes())]
    recovered_img = Image.from_bytes(truncated_recovered_data, hidden_img.shape)
    # recovered_img.save(f'output/{img_name}_recovered_{mode.name}_{num_lsb}bit{with_key}.jpg')


if __name__ == '__main__':
    cover_img = Image.from_file('images/mit_512x384.jpg')
    hidden_img = Image.from_file('images/mit_128x96.jpg')
    message = b'I hate crypto'
    secret_key = b'password'

    # (1) Hide text message inside cover image using LSB
    # stego_img = stego.hide(cover_img, message, mode=stego.Mode.LSB)
    # recovered_data = stego.recover(stego_img, mode=stego.Mode.LSB)
    # truncated_recovered_data = recovered_data[:len(message)]
    # print(truncated_recovered_data.decode('utf-8'))

    # (2) Hide an (unencrypted) image inside cover image using LSB
    # run_stego_once('mit', cover_img, hidden_img, stego.Mode.LSB)

    # (3) Hide an (unencrypted) image inside cover image using DCT + key steganography
    for i in range(1, 5):
        run_stego_once('mit', cover_img, hidden_img, stego.Mode.DCT, key=None, num_lsb=i)

    # run_stego_all('mit', cover_img, hidden_img, secret_key)
