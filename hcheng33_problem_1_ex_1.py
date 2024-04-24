
from skimage import io
from matplotlib import pyplot as plt
from PIL import Image
import cv2
import numpy as np

import plotly.express as px

def part_1a(im_raw):

    # reading tiff file
    #im_raw = io.imread("TEST_MOVIE_00001-small-motion.tif")

    # Note: I broke it down in two halves because the animation time stopped scaling linearly
    # after around 300 frames
    imfirst = im_raw[0:249,:,:]
    imsecond = im_raw[250:499,:,:]

    # animating the tiff data
    fig_first = px.imshow(
        imfirst,
        animation_frame=0,
    )

    fig_second = px.imshow(
        imsecond,
        animation_frame=0,
    )

    fig_first.show()
    fig_second.show()

    return

def part_1b(im_raw):

    #im_raw = io.imread("TEST_MOVIE_00001-small-motion.tif")

    # choosing images with apparent wiggle
    image1 = im_raw[36,:,:]
    image2 = im_raw[38,:,:]

    # cmopute image correlation with opencv filter
    corr = cv2.filter2D(image1, ddepth=-1, kernel=image2, borderType = 0)
    plt.matshow(corr)
    plt.title("Correlation between two frames")
    plt.show()

    # find the peak index
    shift_ind = np.unravel_index(corr.argmax(), corr.shape)
    print("The shift for peak correlation is: ", shift_ind[0]-250, shift_ind[1]-250)

    