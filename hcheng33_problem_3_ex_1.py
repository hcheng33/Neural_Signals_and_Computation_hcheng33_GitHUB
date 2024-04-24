from skimage import io
from matplotlib import pyplot as plt
from PIL import Image
import cv2
import numpy as np

import plotly.express as px


def part_3a(im_raw):

    # using mean frame data
    im_mean = im_raw.mean(axis=0)
    im_mean = np.array(im_mean,dtype="float")
    #print(cv2.THRESH_BINARY)

    # using threshold to create binary mask of ROI
    bin_mask = cv2.threshold(im_mean,600,255,cv2.THRESH_BINARY)

    # plotting binary mask
    plt.matshow(bin_mask[1])
    plt.title("Binary Mask of ROI")
    plt.show()

    # segmenting the identified ROIs
    mask_fg = bin_mask[1].astype(np.uint8) 
    numLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_fg,connectivity=8)

    # locating five ROIs with largest area
    area = stats[:, cv2.CC_STAT_AREA]
    roi = np.argsort(area)[-6:-1]

    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5)
    fig.suptitle('Five Selected ROIs')
        
    ax1.imshow(labels == roi[0])
    ax2.imshow(labels == roi[1])
    ax3.imshow(labels == roi[2])
    ax4.imshow(labels == roi[3])
    ax5.imshow(labels == roi[4])
    plt.show()