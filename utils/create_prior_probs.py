import numpy as np
from paddle.fluid.data import data
from skimage.io import imread
from skimage import color
import random
from skimage.transform import resize
from paddle.vision.datasets import ImageFolder
from paddle.io import random_split, DataLoader
from glob import glob
from os.path import *
import os
import multiprocessing

# Read into file paths and randomly shuffle them
data_dir = "/data"


def prob_loader(path):
    return imread(path)


train_dataset = ImageFolder(os.path.join(data_dir, "train"), loader=prob_loader)
# Load 313 bins on the gamut
# points (313, 2)
global points = np.load('model/pretrain/pts_in_hull.npy')
points = points.astype(np.float64)
# points (1, 313, 2)
points = points[None, :, :]

# probs (313,)
probs = np.zeros((points.shape[1]), dtype=np.float64)
num = 0
pool = multiprocessing.Pool(processes=64)


def get_index(in_data):
    expand_in_data = np.expand_dims(in_data, axis=1)
    distance = np.sum(np.square(expand_in_data - points), axis=2)

    return np.argmin(distance, axis=1)


def cal_image(img)


for num, img in enumerate(train_dataset):
    # img = imread(img_f)
    img = resize(img[0], (256, 256), preserve_range=True)

    # Make sure the image is rgb format
    if len(img.shape) != 3 or img.shape[2] != 3:
        continue
    img_lab = color.rgb2lab(img)
    img_lab = img_lab.reshape((-1, 3))

    # img_ab (256^2, 2)
    img_ab = img_lab[:, 1:].astype(np.float64)

    nd_index = get_index(img_ab)
    for i in nd_index:
        i = int(i)
        probs[i] += 1

# Calculate probability of each bin
probs = probs / np.sum(probs)
# print(probs)
# Save the result
print(np.sum(probs))
np.save('sintel_ctest_prior_probs', probs)
