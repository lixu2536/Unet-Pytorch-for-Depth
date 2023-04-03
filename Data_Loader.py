from __future__ import print_function, division
import os

import cv2
from PIL import Image
import torch
import torch.utils.data
import torchvision
from skimage import io
from torch.utils.data import Dataset
import random
import numpy as np


class Images_Dataset(Dataset):
    """Class for getting data as a Dict
    Args:
        images_dir = path of input images
        labels_dir = path of labeled images
        transformI = Input Images transformation (default: None)
        transformM = Input Labels transformation (default: None)
    Output:
        sample : Dict of images and labels"""

    def __init__(self, images_dir, labels_dir, transformI = None, transformM = None):

        self.labels_dir = labels_dir
        self.images_dir = images_dir
        self.transformI = transformI
        self.transformM = transformM

    def __len__(self):
        return len(self.images_dir)

    def __getitem__(self, idx):

        for i in range(len(self.images_dir)):
            image = io.imread(self.images_dir[i])
            label = io.imread(self.labels_dir[i])
            if self.transformI:
                image = self.transformI(image)
            if self.transformM:
                label = self.transformM(label)
            sample = {'images': image, 'labels': label}

        return sample


class Images_Dataset_folder(torch.utils.data.Dataset):
    """Class for getting individual transformations and data
    Args:
        images_dir = path of input images
        labels_dir = path of labeled images
        transformI = Input Images transformation (default: None)
        transformM = Input Labels transformation (default: None)
    Output:
        tx = Transformed images
        lx = Transformed labels"""

    def __init__(self, images_dir, labels_dir,transformI = None, transformM = None):
        self.images = sorted(os.listdir(images_dir))
        self.labels = sorted(os.listdir(labels_dir))  # 进行人为排序！ 指定个数
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transformI = transformI
        self.transformM = transformM

        if self.transformI:
            self.tx = self.transformI
        else:
            self.tx = torchvision.transforms.Compose([
               # torchvision.transforms.Resize((128, 128)),  # 《《《《《96->128 数据处理，影响处理速度
               #   torchvision.transforms.CenterCrop(128), ###
               #   torchvision.transforms.RandomRotation((-10,10)),  #####  # 默认围绕中心进行-10，10角度的随机旋转
               # torchvision.transforms.RandomHorizontalFlip(),
               torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),   # 亮度，对比度，饱和度
               torchvision.transforms.ToTensor(),
               # torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
               torchvision.transforms.Normalize(mean=[0.081068], std=[0.184901])  # 尝试反归一化
            ])

        if self.transformM:
            self.lx = self.transformM
        else:
            self.lx = torchvision.transforms.Compose([
                # torchvision.transforms.Resize((128, 128)),
                #   torchvision.transforms.CenterCrop(128),
                #   torchvision.transforms.RandomRotation((-10, 10)),
                # torchvision.transforms.Grayscale(),  # 对标签转为灰度图像 浮点数据 取整了；不转化灰度，totensor不处理（/255）
                torchvision.transforms.ToTensor(),      # 转为灰度后 从 [0, 255] 归一化到[0,1]
                # torchvision.transforms.Normalize(mean=[0.5], std=[0.5])  # 当使用二元交叉熵loss时，进行标签归一化
                # torchvision.transforms.Lambda(lambda x: torch.cat([x, 1 - x], dim=0))
            ])

    def __len__(self):

        return len(self.images)

    def __getitem__(self, i):
        i1 = Image.open(self.images_dir + self.images[i])
        # l1 = Image.open(self.labels_dir + self.labels[i])   # 加载 float32 tiff 报错
        l1 = cv2.imread(self.labels_dir + self.labels[i], -1)   # cv2读取tiff，数据为numpy.ndarray
        ##############################
        # npy = np.load(self.labels_dir)      # 使用npy文件直接读入，判断是否由于图片过大导致的显存不足。
        # con = npy[i, :, :, :]
        # con = np.reshape(con, (480, 640))
        # l1 = np.asarray(con, np.float32)
        ###############################
        # l1 = (l1-l1.min())/(l1.max()-l1.min())    # 深度标准化：(l-min)/(max-min) [0-1] 浮点数据训练结果不正常，继续采用！ 数据集
        # l1 = l1/255    # 深度变化，/255，弥补tensor的不处理 （是否可以处理为上[0-1]）
        l1 = (l1-22.294105)/249.756889      # 对深度图的归一化处理，采用标签数据集的min，max
        l1 = Image.fromarray(l1)    # 将cv格式转为pil格式，用于后续的transform

        # seed = np.random.randint(0, 2**32, dtype=np.uint32)  # make a seed with numpy generator 修改类型界限

        # apply this seed to img tranfsorms
        # random.seed(seed)
        # torch.manual_seed(seed)
        img = self.tx(i1)
        
        # apply this seed to target/label tranfsorms  
        # random.seed(seed)
        # torch.manual_seed(seed)
        label = self.lx(l1)
        # label = label.cpu().detach().numpy()  # tensor后，debug查看转为array
        # label = np.asarray(label)   # 不tensor为image，debug转为array

        return img, label

