import os
import sys
import six
import re
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image,ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from dataload.dataAug import *
import torchvision.transforms as transforms
from utils.label2tensor import strLabelConverter

# 文本不使用lmdb 直接从txt文件加载
# 对图像进行padding
def Add_Padding(image, top, bottom, left, right, color=(255,255,255)):
    if(not isinstance(image,np.ndarray)):
        image = np.array(image)
    padded_image = cv2.copyMakeBorder(image, top, bottom,left, right, cv2.BORDER_CONSTANT, value=color)
    return padded_image

# 将中文字符进行join
def fixkeyCh(key):
    return ''.join(re.findall("[㙟\u4e00-\u9fa50-9a-zA-Z#%().·-]",key))

# 英文字符进行join
def fixkeyEn(key):
    return ''.join(re.findall("[0-9a-zA-Z]",key))


# 基于lmdb_file 完成数据读取
class LoadDataset(Dataset):
    def __init__(self, config, train=True):
        num_workers = config['train']['num_workers']
        self.fixKey = config['train']['fixKeyON']
        self.fixKeyType = config['train']['fixKeytype']
        assert self.fixKeyType in ['En', 'Ch']
        self.picpaths, self.labels = self.readlines()
        if train:
            self.picpaths = self.picpaths[100000:]
            self.labels = self.labels[100000:]
        else:
            self.picpaths = self.picpaths[:100000]
            self.labels = self.labels[:100000]
        self.buffer = self.getbuffer()

    def readlines(self):
        srcdir = "/code/wangshiyuan02/data/ocr/commocr/textline/generateData/companyname700w40032"
        txtpath = "/code/wangshiyuan02/data/ocr/commocr/textline/generateData/tmp_labels700w.txt"
        picpaths = []
        labels = []
        with open(txtpath, "r", encoding="utf-8") as f:
            line = f.readline()
            while line:
                if "_" not in line:
                    line = f.readline()
                    continue
                factors = line.strip().split()
                picname = factors[0]+".jpg"
                label = factors[-1]
                picpath = os.path.join(srcdir, picname)
                if os.path.exists(picpath):
                    picpaths.append(picpath)
                    labels.append(label)
                line = f.readline()
        return picpaths, labels

    def getbuffer(self):
        picpath = self.picpaths[0]
        label = self.labels[0]
        img = Image.open(picpath).convert('RGB')
        return (img, label)

    def __len__(self):
        return len(self.picpaths)

    def __getitem__(self, index):
        try:
            picpath = self.picpaths[index]
            label = self.labels[index]
            img = Image.open(picpath).convert('RGB')
            return (img, label)
        except:
            return self.buffer


class resizeNormalize(object):
    def __init__(self, height=32, max_width=280, types='train'):
        assert types in ['train', 'val', 'test']
        self.toTensor = transforms.ToTensor()
        self.max_width = max_width
        self.types = types
        self.height = height

    def __call__(self, img):
        if (self.types == 'train' or self.types == 'val'):
            w, h = img.size
            img = img.resize((int(self.height / float(h) * w), self.height), Image.BILINEAR)
            w, h = img.size
            if (w < self.max_width):
                img = Add_Padding(img, 0, 0, 0, self.max_width - w)
                img = Image.fromarray(img)
            else:
                img = img.resize((self.max_width, self.height), Image.BILINEAR)
        elif self.types == 'test':
            w, h = img.size
            img = img.resize((int(self.height / float(h) * w) // 4 * 4, self.height), Image.BILINEAR)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img


# 获取图像、标签tensor 的函数
class alignCollate(object):
    def __init__(self, config, trans_type):
        self.imgH = config['train']['imgH']
        self.imgW = config['train']['imgW']
        self.use_tia = config['train']['use_tia']
        self.aug_prob = config['train']['aug_prob']
        self.label_transform = strLabelConverter(config['train']['alphabet'])
        self.trans_type = trans_type
        self.isGray = config['train']['isGray']
        self.ConAug = config['train']['ConAug']

    def __call__(self, batch):
        images, labels = zip(*batch)
        new_images = []
        for (image, label) in zip(images, labels):
            if self.trans_type == 'train':

                #                 image = np.array(image)
                #                 try:
                #                     image = warp(image,self.use_tia,self.aug_prob)
                #                 except:
                #                     pass
                #                 image = Image.fromarray(image)

                if self.isGray:
                    image = image.convert('L')
            new_images.append(image)
        transform = resizeNormalize(self.imgH, self.imgW, self.trans_type)

        fix_image = []
        fix_label = []
        for (img, label) in zip(new_images, labels):
            try:
                img = transform(img)
                fix_image.append(img)
                fix_label.append(label)
            except:
                pass
        fix_image = torch.cat([t.unsqueeze(0) for t in fix_image], 0)
        intText, intLength = self.label_transform.encode(fix_label)
        return fix_image, intText, intLength, fix_label


def CreateDataset(config, lmdb_type):
    assert lmdb_type in ['train', 'val']
    if lmdb_type == 'train':
        train_dataset = LoadDataset(config)
        return train_dataset
    elif lmdb_type == 'val':
        val_datasets = LoadDataset(config, False)
        return val_datasets

