

from typing import List
import cv2
import numpy as np
import matplotlib.pyplot as plt

from image import Image
import steganography as stego
import metrics as metriclib
from metrics import Metric

def plot_psnr_graph(cover_img: cv2.Mat, full_secret: bytes, file_to_save: str, hidden_capacity_perc: int = 5) -> None:
    """
    Save psnr plot for given cover_img for different hiding
    if cover_img is not grayscale, will be using grayscale
    """

    # assume using grayscale for now

    size = np.prod(cover_img.shape)//8
    perc_increment = hidden_capacity_perc
    stego_modes = [stego.Mode.LSB, stego.Mode.DCT]
    mode_to_psnr = {stego_method:[] for stego_method in stego_modes}
    x = []

    # for pvd
    int_key = 1000
    key = int_key.to_bytes(2, 'big')

    for hidden_capacity in range(50, size, size*perc_increment//100):
        for stego_method in mode_to_psnr:
            secret = full_secret[:hidden_capacity]
            stego_img = stego.hide(Image(cover_img), secret, stego_method).matrix if hidden_capacity > 0 else cover_img

            if stego_method == stego.Mode.PVD:
                print(stego.recover(stego_img, stego_method, key=key))
            mode_to_psnr[stego_method].append(metriclib.get_psnr(cover_img, stego_img))
        x.append(hidden_capacity/size*100)

    # plotting
    fig, ax = plt.subplots()

    title = stego_method.name if stego_method else 'COVER IMAGE'
    ax.set_title(f'PSNR Plot for Different Stego Methods')
    ax.set_ylabel('PSNR')
    ax.set_xlabel('Percentage of hiding capacity')
    for stego_method in mode_to_psnr:
        ax.plot(x, mode_to_psnr[stego_method], label=stego_method.name)
    ax.legend()
    plt.savefig(file_to_save)


def plot_rs_graph(cover_img: cv2.Mat, full_secret: bytes, file_to_save: str, hidden_capacity_perc: int = 5, stego_method = None) -> None:
    """
    Save rs plot for the given cover_img storing full_secret at different capacities
    if cover_img is not grayscale, will be using grayscale
    """

    # assume using grayscale for now
    if len(cover_img.shape) == 3:
        cover_img = cover_img[:, :, 0]

    # TODO: update metrics to consider the size in bytes vs bits!
    size = np.prod(cover_img.shape)//8
    perc_increment = hidden_capacity_perc
    metrics = {Metric.SM: [], Metric.RM: [], Metric.RM_NEG: [], Metric.SM_NEG: []}
    x = []



    for hidden_capacity in range(0, size, size*perc_increment//100):
        secret = full_secret[:hidden_capacity]
        img = cover_img
        if stego_method:
            img = stego.hide(Image(cover_img), secret, mode=stego_method).matrix  # NOTE: i changed the API; will tidy up later
        elif hidden_capacity == 0:
            img = cover_img.astype(np.float64)
        rm, sm, rm_neg, sm_neg = metriclib.get_rs(img)

        metrics[Metric.RM].append(rm)
        metrics[Metric.SM].append(sm)
        metrics[Metric.RM_NEG].append(rm_neg)
        metrics[Metric.SM_NEG].append(sm_neg)

        x.append(hidden_capacity / size)

    # plotting
    fig, ax = plt.subplots()

    title = stego_method.name if stego_method else 'COVER IMAGE'
    ax.set_title(f'RS Plot for {title}')
    ax.set_ylabel('Percentage of the regular and singular pixel groups')
    ax.set_xlabel('Percentage of hiding capacity')
    for metric in metrics:
        ax.plot(x, metrics[metric], label=metric.name)
    ax.legend()
    plt.savefig(file_to_save)


if __name__ == "__main__":
    # Get inputs
    cover_img = Image.from_file('images/testimage1_128x96.jpg')
    secret_text = ''
    with open('secret.txt', 'rb') as f:
        secret_text = f.read()

    img_types = ['charlesriver', 'mit', 'testimage1']
    modes = [stego.Mode.LSB, stego.Mode.DCT]

    # for img_type in img_types:
    #     for mode in modes:
    #         stego_img_array = Image.from_file(f"output/{img_type}_stego_{mode.name}.jpg").matrix
    #         cover_img_array = Image.from_file(f"images/{img_type}_512x384.jpg").matrix

    #         print(f"PSNR for {img_type} in mode {mode.name} = {metriclib.get_psnr(cover_img_array, stego_img_array)}")
    plot_rs_graph(cover_img.matrix, secret_text, 'rs_plot_LSB.png', stego_method=stego.Mode.LSB)
    plot_rs_graph(cover_img.matrix, secret_text, 'rs_plot_orig.png')
    plot_rs_graph(cover_img.matrix, secret_text, 'rs_plot_DCT.png', stego_method=stego.Mode.DCT)
    plot_psnr_graph(cover_img.matrix, secret_text, 'psnr_plot.png')
