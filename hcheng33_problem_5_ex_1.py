import numpy as np
import pandas as pd
from skimage import io
from matplotlib import pyplot as plt
import scipy
import cv2
from PIL import Image

from sklearn.decomposition import PCA, NMF, FastICA

def part_5a(im_raw):

    im_vec = im_raw.reshape(im_raw.shape[0], -1)
    pca = PCA(n_components=40)
    pca.fit_transform(im_vec)

    plt.plot(np.cumsum(pca.explained_variance_ratio_),marker="*")
    plt.title("Cumulative Variance")
    plt.xlabel("# of components")
    plt.ylabel("Cumulative Variance")
    plt.show()

    fig, axs = plt.subplots(1,4)
    fig.suptitle("PCA Components")

    for i in range(0,4):
        axs[i].imshow(pca.components_[i].reshape(491,491))
    plt.show()

def part_5b(im_raw):

    im_vec = im_raw.reshape(im_raw.shape[0], -1)

    COM = range(2,15)
    err = []

    for com in COM:

        # plotting error of different component numbers
        nmf = NMF(n_components=com)
        nmf.fit_transform(im_vec)

        err.append(nmf.reconstruction_err_)

    plt.plot(COM,err)
    plt.title("NMF Error for Different Number of Components")
    plt.show()

    # choose the component number with smallest error and plot components
    nmf = NMF(n_components=4)
    nmf.fit_transform(im_vec)


    fig, axs = plt.subplots(1,4)
    fig.suptitle("NMF Components")

    for i in range(0,4):
        axs[i].imshow(nmf.components_[i].reshape(491,491))
    plt.show()

def part_5c(im_raw):

    im_vec = im_raw.reshape(im_raw.shape[0], -1)


    err = []
    COM = range(3,10)

    # plotting error of different component numbers
    for com in COM: 
        ica = FastICA(n_components=com)
        ica.fit(im_vec)

        err.append(np.sum((im_vec - ica.inverse_transform(ica.transform(im_vec)))**2))

    plt.plot(err)
    plt.title("ICA Error for Different Number of Components")


    fig, axs = plt.subplots(1,4)
    fig.suptitle("ICA Components")
    for i in range(0,4):
        axs[i].imshow(ica.components_[i].reshape(491,491))
    plt.show()