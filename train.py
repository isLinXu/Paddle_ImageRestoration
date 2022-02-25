import argparse
import os
from PIL.Image import new
import matplotlib
import paddle
from colorizers import *
from paddle.vision.datasets import ImageFolder
import paddle.fluid as fluid
from paddle.io import random_split, DataLoader
from skimage import color
import time
import pdb
import paddle.nn as nn
from paddle.vision.models import vgg16
import cv2

#import paddle.distributed as dist
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
parser.add_argument("--data_dir", type=str, default="/home/linxu/Desktop/GAN/Paddle-Colorization/data", help="dataset with train/val/test set")
parser.add_argument("-bs","--batch_size", type=int, default=32)
parser.add_argument("-e","--epochs", type=int, default=32)
parser.add_argument("--save_dir", type=str, default="model/")
parser.add_argument("--pretrain", action="store_true", help="whether to use pretrain model")


opt = parser.parse_args()
model = eccv16(pretrained=opt.pretrain)
if opt.use_gpu:
    paddle.set_device('gpu')
if opt.epochs > 0:
    #train and eval
    train_dataset = ImageFolder(os.path.join(opt.data_dir, "train"), loader=lab_loader)
    ori_val_dataset = ImageFolder(os.path.join(opt.data_dir, "val"), loader=lab_loader)
    print("length of val dataset :", len(ori_val_dataset))
    val_dataset, _ = random_split(ori_val_dataset, [10000,len(ori_val_dataset)-10000])
    scheduler = paddle.optimizer.lr.PiecewiseDecay(boundaries=[2e5, 375e3], 
                                                values=[3e-5, 1e-5, 3e-6], 
                                                )
    adam = paddle.optimizer.Adam(learning_rate=scheduler,
                                beta1=0.9, 
                                beta2=0.99,
                                weight_decay=1e-3,
                                parameters=model.parameters())

    train_loader = DataLoader(train_dataset,
                        #places = place,
                        batch_size=opt.batch_size,
                        shuffle=True,
                        drop_last=True,
                        num_workers=32)

    val_loader = DataLoader(val_dataset,
                        #places = place,
                        batch_size=opt.batch_size,
                        shuffle=False,
                        drop_last=True,
                        num_workers=32)
    criterion = nn.CrossEntropyLoss(soft_label=True)
    # Training session
    if not os.path.exists(opt.save_dir):
        os.makedirs(opt.save_dir)
    lowest_loss = 1e5
    losses = []
    model.train()
    for e in range(opt.epochs):  
        train_loss = []
        train_start = time.time()
        epoch_start = time.time()
        for i, batch in enumerate(train_loader):
            out, target = model(batch[0][0])   # batch[0][1]
            loss = criterion(out, target)
            loss.backward()  
            adam.step()
            scheduler.step()
            adam.clear_grad()
            train_loss.append(loss.numpy())
            if i % 100== 0:
                training_time = time.time()-train_start
                train_start = time.time()
                print('iter:%d  time:%d  avg_loss%.4f'%(i,training_time,np.mean(train_loss[-100:])))
        print('Time per epoch:%.2f,loss:%.4f'%(time.time()-train_start,np.mean(train_loss)))
        
        if e % 1 == 0:
            paddle.save(model.state_dict(),f'model/ckpt/training_{e}')
            model.eval()
            eval_loss = []
            eval_start = time.time()
            for i, batch in enumerate(val_loader):
                out, target = model(batch[0][0],batch[0][1])   
                loss = criterion(out, target)
                loss.backward()  
                adam.step()
                scheduler.step()
                adam.clear_grad()
                eval_loss.append(loss.numpy())
                if np.mean(eval_loss) < lowest_loss:
                    lowest_loss = np.mean(eval_loss)
                    paddle.save(model.state_dict(),f'model/pretrain/best.pdparams')  # save for training 
            model.train()
            print('Time per epoch:%.2f,loss:%.4f, lowest loss:%.4f'%(time.time()-eval_start,np.mean(eval_loss), lowest_loss))
else:
    # Eval session
    classifier = vgg16(pretrained=True)
    model.eval()
    classifier.eval()
    valImageLoader = val_loader("model/pretrain/val.txt")
    test_origin_dataset = ImageFolder(os.path.join(opt.data_dir, "val"), loader=valImageLoader.load)
    test_vgg_dataset = paddle.io.Subset(dataset=test_origin_dataset, indices=range(5000,5500))
    test_vgg_loader = DataLoader(test_vgg_dataset,
                        # places = place,
                        batch_size=1,#
                        shuffle=True,
                        drop_last=True,#
                        num_workers=8)

    m = paddle.metric.Accuracy()
    m_origin = paddle.metric.Accuracy()
    res = m.accumulate()
    img_mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    img_std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
    for i, batch in enumerate(test_vgg_loader):
        # print(batch)
        new_img = paddle.concat((batch[0][0],batch[0][1]),axis=1)
        new_img = paddle.transpose(new_img,(0,2,3,1))
        new_img = paddle.squeeze(new_img).numpy()
        new_img = color.lab2rgb(new_img)*255
        new_img = cv2.cvtColor(new_img, cv2.COLOR_RGB2BGR)
        new_img = cv2.resize(new_img, (224, 224))
        new_img = new_img.transpose((2,0,1))/255. - img_mean
        new_img = new_img/img_std
        new_img = paddle.to_tensor(new_img, dtype=paddle.float32)
        new_img = paddle.unsqueeze(new_img,0)
        result = classifier(new_img)
        correct = m.compute(result, batch[0][2])
        m.update(correct)
        result_origin = classifier(batch[0][3])
        correct = m.compute(result_origin, batch[0][2])
        m_origin.update(correct)
        if i %100 == 0:
            print(f"finished {i}! Acc = {m.accumulate()}, origin Acc = {m_origin.accumulate()}")
    # print(m.accumulate(), m_origin.accumulate())
    print(f"finished {i}! Acc = {m.accumulate()}, origin Acc = {m_origin.accumulate()}")



    







