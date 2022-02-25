import numpy as np
from paddle.fluid.data import data
from skimage.io import imread
from skimage import color
import random
from skimage.transform import resize
from paddle.vision.datasets import ImageFolder
from paddle.io import random_split, DataLoader
import paddle
from glob import glob
from os.path import *
import os
import multiprocessing
from PIL import Image

# Read into file paths and randomly shuffle them
data_dir = "/data"
paddle.set_device('gpu')
place = paddle.CUDAPlace(0)

def prob_loader(path):
    img = np.asarray(Image.open(path).convert('RGB'))
    img = resize(img, (256, 256), preserve_range=True)
    img_lab = color.rgb2lab(img)
    img_lab = img_lab.reshape((-1, 3))
    img_ab = img_lab[:, 1:].astype(np.float32)
    #print(img_ab.shape)
    return paddle.to_tensor(img_ab,dtype=paddle.float32)

train_dataset = ImageFolder(os.path.join(data_dir, "train"),loader=prob_loader)
train_loader = DataLoader(train_dataset,
                    places = place,
                    batch_size=16,
                    shuffle=True,
                    drop_last=False,
                    num_workers=32)

# Load 313 bins on the gamut
# points (313, 2)
points = np.load('model/pretrain/pts_in_hull.npy')
points = points.astype(np.float32)
# points (1, 313, 2)
#points = points[None, :, :]
points = paddle.to_tensor(points, dtype=paddle.float32, place=place)
# probs (313,)
probs = paddle.to_tensor(np.zeros((points.shape[0]), dtype=np.float32), dtype=paddle.float32, place=place)

def get_index( in_data ):
    in_data = paddle.reshape(in_data,(-1,2))    
    in_data = paddle.unsqueeze(in_data, (1))
    #print(in_data.shape)
    in_data = paddle.tile(in_data,[1,313,1])
    distance = paddle.sum(paddle.square(in_data - points), axis=-1)
    #distance = paddle.reshape(distance,(in_data.shape[0]*in_data.shape[1],313 ))
    res = paddle.argmin(distance, axis=-1)
    res = paddle.nn.functional.one_hot(res,313)
    res = paddle.sum(res, axis=0)
    return res

#def cal_image(img)

for num, img in enumerate(train_loader):
    #img = imread(img_f)

    #Make sure the image is rgb format
    #print(img)
    if len(img[0].shape) != 3 or img[0].shape[-1] != 2:
        print("error dimension")

    nd_index = get_index(img[0])
    probs = probs + nd_index
    if num%1000==0:
        print(num)


# Calculate probability of each bin
probs = probs / paddle.sum(probs)
print(probs)
print(paddle.sum(probs))
#print(probs)
# Save the result
#print(np.sum(probs))
np.save('sintel_ctest_prior_probs', probs.numpy())
