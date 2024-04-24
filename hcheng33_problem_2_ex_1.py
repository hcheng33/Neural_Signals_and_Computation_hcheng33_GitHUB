from skimage import io
from matplotlib import pyplot as plt
from PIL import Image
import cv2
import numpy as np

import plotly.express as px

def part_2a(im_raw):

    im_mean = im_raw.mean(axis=0)
    im_median = np.median(im_raw, axis=0)
    im_var = im_raw.var(axis=0)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

    ax1.matshow(im_mean)
    ax1.set_title("Mean")

    ax2.matshow(im_median)
    ax2.set_title("Median")

    ax3.matshow(im_var)
    ax3.set_title("Variance")

    plt.show()

def part_2b(im_raw):

    im_max = im_raw.max(axis=0)
    im_80 = np.percentile(im_raw,80,axis=0)

    fig, (ax1, ax2) = plt.subplots(1, 2)

    ax1.matshow(im_max)
    ax1.set_title("Max")

    ax2.matshow(im_80)
    ax2.set_title("80-th Percentile")
    
    plt.show()