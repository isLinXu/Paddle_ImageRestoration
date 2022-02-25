import numpy as np
import paddle
import paddle.nn.functional as F
from PIL import Image
from skimage import color
from utils.trainable_layers import NNEncLayer
import pdb
import matplotlib as plt
import os
import cv2
#gt_encoder = NNEncLayer()

def load_img(img_path):
    out_np = np.asarray(Image.open(img_path).convert('RGB'))
    if out_np.ndim == 2:
        out_np = np.tile(out_np[:, :, None], 3)
    return out_np



def resize_img(img, HW=(256, 256), resample=3):
    return np.asarray(Image.fromarray(img).resize((HW[1], HW[0]), resample=resample))


def preprocess_img(img_rgb_orig, HW=(256, 256), resample=3):
    # return original size L and resized L as torch Tensors
    img_rgb_rs = resize_img(img_rgb_orig, HW=HW, resample=resample)

    img_lab_orig = color.rgb2lab(img_rgb_orig)
    img_lab_rs = color.rgb2lab(img_rgb_rs)

    img_l_orig = img_lab_orig[:, :, 0]
    img_l_rs = img_lab_rs[:, :, 0]

    tens_orig_l = paddle.to_tensor(img_l_orig, dtype=paddle.float32).unsqueeze((0, 1))
    tens_rs_l = paddle.to_tensor(img_l_rs, dtype=paddle.float32).unsqueeze((0, 1))

    return (tens_orig_l, tens_rs_l)


def postprocess_tens(tens_orig_l, out_ab, mode="bilinear"):
    # tens_orig_l 	1 x 1 x H_orig x W_orig
    # out_ab 		1 x 2 x H x W

    HW_orig = tens_orig_l.shape[2:]
    HW = out_ab.shape[2:]

    # call resize function if needed
    if HW_orig[0] != HW[0] or HW_orig[1] != HW[1]:
        out_ab_orig = F.interpolate(out_ab, size=HW_orig, mode="bilinear")
    else:
        out_ab_orig = out_ab

    out_lab_orig = paddle.concat((tens_orig_l, out_ab_orig), axis=1)

    return color.lab2rgb(out_lab_orig.cpu().numpy()[0, ...].transpose((1, 2, 0)))

def train_preprocess(img_rgb_orig, HW=(256, 256), resample=3):
    img_rgb_rs = resize_img(img_rgb_orig, HW=HW, resample=resample)
    img_lab_rs = color.rgb2lab(img_rgb_rs)
    # np.save("1.npy",img_lab_rs)
    # exit()
    img_lab_rs = img_lab_rs.transpose((2,1,0))
    img_l_rs = img_lab_rs[0:1,:, :]
    img_ab_rs = img_lab_rs[1:,:, :]
    tens_rs_l = paddle.to_tensor(img_l_rs, dtype=paddle.float32)
    tens_rs_ab = paddle.to_tensor(img_ab_rs, dtype=paddle.float32)
    return (tens_rs_l, tens_rs_ab)


def lab_loader(path):
    origin = load_img(path)
    return train_preprocess(origin)

class val_loader(object):
    def __init__(self, cls_path) -> None:
        file = open(cls_path,"r")
        self.cls = {}
        self.img_mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
        self.img_std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
        for line in file.readlines():
            line = line.strip()
            k = line.split(' ')[0]
            v = line.split(' ')[1]
            self.cls[k] = int(v)

    def val_preprocess(self, img_rgb_orig, HW=(256, 256), resample=3,label=-1):
        img_rgb_rs = resize_img(img_rgb_orig, HW=HW, resample=resample)
        img_lab_rs = color.rgb2lab(img_rgb_rs)
        # np.save("1.npy",img_lab_rs)
        # exit()
        img_rgb_rs = cv2.cvtColor(img_rgb_rs, cv2.COLOR_RGB2BGR)
        img_rgb_rs = cv2.resize(img_rgb_rs, (224,224))
        img_lab_rs = img_lab_rs.transpose((2,0,1))
        img_rgb_rs = img_rgb_rs.transpose((2,0,1))/255. - self.img_mean
        img_rgb_rs = img_rgb_rs/self.img_std
        #print(img_lab_rs.shape)
        img_l_rs = img_lab_rs[0,:, :]
        img_ab_rs = img_lab_rs[1:,:, :]
        tens_rs_l = paddle.to_tensor(img_l_rs, dtype=paddle.float32).unsqueeze(0)
        tens_rs_ab = paddle.to_tensor(img_ab_rs, dtype=paddle.float32)
        return (tens_rs_l, tens_rs_ab, paddle.to_tensor([label] ),paddle.to_tensor(img_rgb_rs,dtype=paddle.float32))

    def load(self, path):
        label = self.cls[os.path.basename(path)]
        origin = load_img(path)
        return self.val_preprocess(origin, label=label)





