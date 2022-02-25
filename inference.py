import argparse
import os
#import matplotlib.pyplot as plt
import paddle
from colorizers import *
from paddle.vision.datasets import ImageFolder
import paddle.fluid as fluid
from paddle.io import random_split, DataLoader
import time
import pdb
import paddle.nn as nn
from paddle.vision.models import vgg16

parser = argparse.ArgumentParser()
# parser.add_argument("-i", "--img_path", type=str, default="imgs/ansel_adams3.jpg")
parser.add_argument("--use_gpu", action="store_true", help="whether to use GPU")
parser.add_argument(
    "-o",
    "--save_prefix",
    type=str,
    default="saved",
    help="will save into this file with {eccv16.png, siggraph17.png} suffixes",
)
parser.add_argument("--model_name", type = str, default="eccv16")
parser.add_argument("--data_dir", type=str, default="/data", help="dataset with train/val/test set")
parser.add_argument("-bs","--batch_size", type=int, default=32)
parser.add_argument("-e","--epochs", type=int, default=32)
parser.add_argument("--save_dir", type=str, default="model/")
parser.add_argument("--pretrain", action="store_true", help="whether to use pretrain model")

#os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4"
opt = parser.parse_args()

test_origin_dataset = ImageFolder(os.path.join(opt.data_dir, "val"))
test_vgg_dataset, _ = random_split(test_origin_dataset, [10000,len(test_origin_dataset)-10000])
test_vgg_loader = DataLoader(test_vgg_dataset,
                    #places = place,
                    batch_size=opt.batch_size,
                    shuffle=True,
                    drop_last=True,
                    num_workers=4)
classifier = vgg16(pretrained=True)
classifier.eval()
if opt.model_name == "eccv16":
    model = eccv16(pretrained=opt.pretrain)
model.eval()
for i, batch in enumerate(test_vgg_loader):
    pdb.set_trace()
    out, target = model(batch[0],batch[1])