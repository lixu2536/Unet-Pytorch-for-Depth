import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


# 注意编码方式
def readnpy(file):
    NPY = np.load(file, allow_pickle=True, encoding="latin1")

    print("------data-------")
    print(NPY)
    print("------type-------")
    print(type(NPY))
    print("------shape-------")
    print('ndarray的形状: ', NPY.shape)
    print('ndarray的维度: ', NPY.ndim)
    print('ndarray的元素数量: ', NPY.size)
    print('ndarray中的数据类型: ', NPY.dtype)


def readout(file):
    out = cv2.imread(file, -1)
    # imread:1 彩色，0 灰色，-1 不改变，4？？
    # out = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)

    print("------data-------")
    print(out)
    print("------type-------")
    print(type(out))
    print("------shape-------")
    print('ndarray的形状: ', out.shape)
    print('ndarray中的数据类型: ', out.dtype)
    print('ndarray的维度: ', out.ndim)
    print('ndarray的元素数量: ', out.size)
    return out


def readtiff(inputimg):
    # out = Image.open(inputimg)
    out = cv2.imread(inputimg, -1)
    print("------data-------")
    print(out)
    print("------type-------")
    print(type(out))
    print("------shape-------")
    print('ndarray的形状: ', out.shape)
    print('ndarray中的数据类型: ', out.dtype)
    print('ndarray的维度: ', out.ndim)
    a = out.max()
    print('ndarray的元素数量: ', out.size)
    return out


if __name__ == "__main__":
    # file = "Z_test.npy"
    # readnpy(file)

    # png = "./model-save/model--mse_loss--rmse_score--deep0png/gen_images/im_epoch_5int_0_img_no_1.png"
    # im = readout(png)

    # png = "./model/label_threshold/im_epoch_5int_0_img_no_2.png"
    # im = readout(png)

    # png = "./model/gen_images/im_epoch_20int_0_img_no_1.png"
    # png = "./model/label_threshold/im_epoch_20int_0_img_no_2.png"
    # png = "./model/pred_threshold/im_epoch_20int_0_img_no_2.png"
    # png = "./data/deep22/train/masks/000000.png"
    # im = readout(png)


    png = "./data/tiff5back/train/masks/000324.tiff"
    im = readtiff(png)
