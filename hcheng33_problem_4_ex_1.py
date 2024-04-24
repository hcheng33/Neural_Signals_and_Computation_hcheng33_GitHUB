import numpy as np
import pandas as pd
from skimage import io
from matplotlib import pyplot as plt
import scipy
import cv2
from PIL import Image

import plotly.express as px

def part_4a(im_raw):

    im_mean = im_raw.mean(axis=0)
    im_mean = np.array(im_mean,dtype="float")
    #print(cv2.THRESH_BINARY)

    # using threshold to create binary mask of ROI
    bin_mask = cv2.threshold(im_mean,600,255,cv2.THRESH_BINARY)

    # segmenting the identified ROIs
    mask_fg = bin_mask[1].astype(np.uint8) 
    numLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_fg,connectivity=8)

    # locating five ROIs with largest area
    area = stats[:, cv2.CC_STAT_AREA]
    roi = np.argsort(area)[-6:-1]

    fig, axs = plt.subplots(1,5)
    fig.suptitle("Time Traces of ROIs")
    
    for i in range(0,5):
        roi_select = np.array((labels == roi[i]))
        time_trace = im_raw[:,roi_select].mean(axis=1)

        axs[i].plot(time_trace)

    plt.show()