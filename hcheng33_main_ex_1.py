import numpy as np
import pandas as pd
from skimage import io
from matplotlib import pyplot as plt
import scipy
import cv2
from PIL import Image

from sklearn.decomposition import PCA, NMF, FastICA

import plotly.express as px
import hcheng33_problem_1_ex_1
import hcheng33_problem_2_ex_1
import hcheng33_problem_3_ex_1
import hcheng33_problem_4_ex_1
import hcheng33_problem_5_ex_1

im_raw = io.imread("TEST_MOVIE_00001-small-motion.tif")

## Problem 1
# Part A
hcheng33_problem_1_ex_1.part_1a(im_raw)

# Part B
hcheng33_problem_1_ex_1.part_1b(im_raw)
#The peak correlation is shifted by -5 pixels on the x-axis and -6 pixels on the y-axis 

## Problem 2
# Part A
hcheng33_problem_2_ex_1.part_2a(im_raw)
# The mean and median are blurry likely due to the noisy background activity, the variance is clearer in showing several visible cells

# Part B
hcheng33_problem_2_ex_1.part_2b(im_raw)
# I think a good statistic would clearly segment the cells from background noise. 
# I also tried the max and 80th-percentile for the image. The max showed satisfactory results becasue it locates where the activity is most acitve,
# 80-th percentile was slighly more clear than mean because the noise were a little more supressed

## Problem 3
# Part A
hcheng33_problem_3_ex_1.part_3a(im_raw)

# Part B
# The most intuitive way is to compare it to human eye judgement as the ground truth
# We can also compare the ROI identified to more sophisitcated segmentation algorithms

## Problem 4
# Part A
hcheng33_problem_4_ex_1.part_4a(im_raw)
# I used the connectedComponents function from opencv to identify the ROIs.
# selecting the five ROIs with the largest area, using their pixel index,
# I looked across the average intensity of the time samples within the same ROI pixels

# Part B
# I think it is a little hard to tell for a specific ROI under such a noisy background,
# a better way might be to segment out only the ROI to see it play through time and compare this
# visual info with the intensity trace

## Problem 5
# Part A
hcheng33_problem_5_ex_1.part_5a(im_raw)
# The variance is explained more by the first four principal components, 
# with the subsequent components providing imcremental increase in the explanation of variance

# Part B
hcheng33_problem_5_ex_1.part_5b(im_raw)
# There seems to be an "optimal" number of components that provides the best error results
# aside from that it reverts back to the trend of decreasing error with increased components

# Part C
hcheng33_problem_5_ex_1.part_5c(im_raw)
# The error trend compared with NMF seems to be a steady decrease as number of components increase
# the components plot seems to show for it to be picking up more background noise