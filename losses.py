from __future__ import print_function, division

import torch
import torch.nn.functional as F
from math import exp
import numpy as np
from torch import nn


# 计算一维的高斯分布向量
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)]).cuda()
    return gauss / gauss.sum()


# 创建高斯核，通过两个一维高斯分布向量进行矩阵乘法得到
# 可以设定channel参数拓展为3通道
def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


# 计算SSIM
# 直接使用SSIM的公式，但是在计算均值时，不是直接求像素平均值，而是采用归一化的高斯核卷积来代替。
# 在计算方差和协方差时用到了公式Var(X)=E[X^2]-E[X]^2, cov(X,Y)=E[XY]-E[X]E[Y].
# 正如前面提到的，上面求期望的操作采用高斯核卷积代替。
def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()   # 原带有batch_size时
    # (channel, height, width) = img1.size()      # 经过5back处理后
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret


# Classes to re-use window
class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range

        # Assume 1 channel for SSIM
        self.channel = 1
        self.window = create_window(window_size)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()    # 原带有batch_size时
        # (channel, _, _) = img1.size()      # 经过5back处理后

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel

        return ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)


def SSIM_loss(prediction, target):
    ssim = SSIM()
    loss = 1 - ssim(prediction, target)
    return loss


def ssim_5back(y_predict, y_true):
    # # 去除背景 根据batch_size进行选取图片
    # if y_predict.shape[0] == 2:
    #     pre_bs1 = tensor_mask(y_predict[0, :, :, :], y_true[0, :, :, :])
    #     pre_bs2 = tensor_mask(y_predict[1, :, :, :], y_true[1, :, :, :])
    #     y_predict = torch.stack((pre_bs1, pre_bs2), dim=0).float()
    # global loss_

    global pre_bs1
    global pre_bs2
    global length1
    global length2
    y_true1 = y_true[0].clone()
    y_true2 = y_true[1:2].clone()
    y_true2 = torch.squeeze(y_true2, dim=0)
    y_pred1 = y_predict[0].clone()
    y_pred2 = y_predict[1:2].clone()
    y_pred2 = torch.squeeze(y_pred2, dim=0)
    if y_predict.shape[0] == 2:
        pre_bs1, length1 = torchmask(y_pred1, y_true1)
        pre_bs2, length2 = torchmask(y_pred2, y_true2)

    ssim = SSIM()
    loss = ((1 - ssim(pre_bs1, y_true1)) + (1 - ssim(pre_bs2, y_true2)))/2
    return loss


def ssim_mse(prediction, target, weight=0.8):   # 这种使用方式 梯度传播不会出现叶子节点的问题 无需backward（retain_grad）
    loss1 = SSIM_loss(prediction, target)
    loss2 = mse_loss(prediction, target)
    loss = loss1*weight + loss2*(1-weight)
    return loss

def dice_loss(prediction, target):
    """Calculating the dice loss
    Args:
        prediction = predicted image
        target = Targeted image
    Output:
        dice_loss"""

    smooth = 1.0

    i_flat = prediction.view(-1)
    t_flat = target.view(-1)

    intersection = (i_flat * t_flat).sum()

    return 1 - ((2. * intersection + smooth) / (i_flat.sum() + t_flat.sum() + smooth))


def calc_loss(prediction, target, bce_weight=0.5):
    """Calculating the loss and metrics
    Args:
        prediction = predicted image
        target = Targeted image
        metrics = Metrics printed
        bce_weight = 0.5 (default) bce:二元交叉熵，logits方式自带归一化，更好的数据稳定性
    Output:
        loss : dice loss of the epoch """
    bce = F.binary_cross_entropy_with_logits(prediction, target)  # 不需要sigmoid，自动会进行
    prediction = torch.sigmoid(prediction)
    # prediction = torch.relu(prediction)
    dice = dice_loss(prediction, target)

    loss = bce * bce_weight + dice * (1 - bce_weight)
    return loss


def mse_5backloss(y_predict, y_true):
    mask = y_true == -1
    loss_fn = torch.nn.MSELoss(reduction='mean')
    loss = loss_fn(y_predict[~mask], y_true[~mask])
    return loss


# def mse_5backloss(y_predict, y_true):
#     # # 去除背景 根据batch_size进行选取图片
#     # if y_predict.shape[0] == 2:
#     #     pre_bs1 = tensor_mask(y_predict[0, :, :, :], y_true[0, :, :, :])
#     #     pre_bs2 = tensor_mask(y_predict[1, :, :, :], y_true[1, :, :, :])
#     #     y_predict = torch.stack((pre_bs1, pre_bs2), dim=0).float()
#
#         # global pre_bs1
#         global pre_bs2
#         global length1
#         global length2
#         y_true1 = y_true[0].clone()
#         y_true2 = y_true[1:2].clone()
#         y_true2 = torch.squeeze(y_true2, dim=0)
#         y_pred1 = y_predict[0].clone()
#         y_pred2 = y_predict[1:2].clone()
#         y_pred2 = torch.squeeze(y_pred2, dim=0)
#         if y_predict.shape[0] == 2:
#             pre_bs1, length1 = torchmask(y_pred1, y_true1)
#             pre_bs2, length2 = torchmask(y_pred2, y_true2)
#         # new_predict = torch.stack((pre_bs1, pre_bs2), dim=0).float()    # 两个new展开一维后维度不同
#         # new_true = torch.stack((true1, true2), dim=0).float()
#         loss_mse = (torch.sum(torch.square(pre_bs1 - y_true1)) / length1 +
#                     torch.sum(torch.square(pre_bs2 - y_true2)) / length2)/2
#
#         return loss_mse


def mse_loss(prediction, target):
    mse = torch.nn.MSELoss(reduction='mean')
    loss = mse(prediction, target)
    return loss


def bce_loss(prediction, target):
    bce = F.binary_cross_entropy_with_logits(prediction, target)  # 不需要sigmoid，自动会进行
    loss = bce
    return loss


def L1_loss(prediction, target):
    l1 = torch.nn.L1Loss(reduction='mean')
    loss = l1(prediction, target)
    return loss


def threshold_predictions_v(predictions, thr=150):
    thresholded_preds = predictions[:]
    # hist = cv2.calcHist([predictions], [0], None, [2], [0, 2])
    # plt.plot(hist)
    # plt.xlim([0, 2])
    # plt.show()
    low_values_indices = thresholded_preds < thr
    thresholded_preds[low_values_indices] = 0
    low_values_indices = thresholded_preds >= thr
    thresholded_preds[low_values_indices] = 255
    return thresholded_preds


def threshold_predictions_p(predictions, thr=0.01):
    thresholded_preds = predictions[:]
    # hist = cv2.calcHist([predictions], [0], None, [256], [0, 256])
    low_values_indices = thresholded_preds < thr
    thresholded_preds[low_values_indices] = 0
    low_values_indices = thresholded_preds >= thr
    thresholded_preds[low_values_indices] = 1

    return thresholded_preds


def masktransform(inputimg, maskimg):   # 根据label 对array 进行去除背景

    mask = np.zeros(maskimg.shape)

    rows, cols = mask.shape[:2]
    for row in range(rows):
        for col in range(cols):
            if maskimg[row, col] != 0:
                mask[row, col] = 1

    outimg = mask * inputimg

    return outimg


def tensor_mask(inputimg, maskimg):   # 根据label 对array 进行去除背景
    inputimg = torch.transpose(inputimg, 0, 2)
    maskimg = torch.transpose(maskimg, 0, 2)
    mask = torch.zeros(maskimg.shape).cuda()

    rows, cols = mask.shape[:2]
    for row in range(rows):
        for col in range(cols):
            if maskimg[row, col] != 0:
                mask[row, col] = 1

    outimg = torch.transpose((mask.cuda() * inputimg.cuda()), 0, 2)
    return outimg


# function to create masked array for Numpy array
def masking(ar1, ar2):
    # creating the mask of array2 where
    # condition array2 mod 3 is true
    mask = np.ma.masked_where(ar2 == 0, ar2)

    # getting the mask of the array
    res_mask = np.ma.getmask(mask)

    ar1[res_mask] = 0
    print(ar1)

    # count nonzero element
    cnt = np.count_nonzero(ar1)
    print(cnt)

    # masking the array1 with the result
    # of mask of array2
    masked = np.ma.masked_array(ar1, mask=res_mask)

    return masked


# 创建mask 去除tensor元素 创建新tensor
def torchmask(ar1, ar2):
    """
    input: ar1 = y_pred ; ar2 = y_true shape: [1, 480 ,640]
    output: new_ = y_pred:5back=-1 ; length = important area  pixel number
    """
    ar1 = torch.transpose(ar1, 0, 2)
    ar2 = torch.transpose(ar2, 0, 2)
    new_ar1 = torch.where(ar2 == -1, ar2, ar1)  # ar2符合条件保留，否侧为ar1（2中是-1的位置，在1中替换为-1）
    # print('where: ', where)

    index = (new_ar1 == -1).nonzero()     # 返回符合条件的索引值
    # print('value=-1：', index1)

    # length = index.numel()/4
    length = index.shape[0]
    # print('sum1:', length1)

    # new_pred = torch.cat([torch.cat((new_ar1[i][0:j], new_ar1[i][j + 1:])) for i, j in index]) # 用torch.cat()删除index位置
    # print(new_, new_.dtype)
    # new_true = torch.cat([torch.cat((ar2[i][0:j], ar2[i][j + 1:])) for i, j in index])  # 用torch.cat()删除index位置
    # print(new_, new_.dtype)
    new_pred = torch.transpose(new_ar1, 0, 2)
    # new_pred = new_ar1
    return new_pred, 307200-length


def new_mask(ar1, ar2):
    index = (ar2 == -1).nonzero()  # 返回背景的索引位置
    # index = torch.LongTensor(index)
    mask = torch.zeros_like(ar2)
    mask = torch.gather(mask, index=index)  # 背景是1，true
    # mask.bool()
    #################################
    # 方法1、根据选择，返回物体位置的一维数组，全展开。
    new_pred = torch.masked_select(input=ar1, mask=mask)    # 新一维数组
    new_true = torch.masked_select(input=ar2, mask=~mask)

    # ##############################
    # # 方法2、根据mask 将背景命为-1
    # new_pred = torch.masked_fill(input=ar1, mask=mask, value=-1)
    #
    # # ##############################
    # # 方法3、将mask的数据赋值给source
    # source = torch.zeros_like(ar1)
    # new_pred = torch.masked_scatter(input=ar1, mask=~mask, source=source)   # 将物体的像素选取出来

    length = index.shape[0]  # 背景像素个数
    # print('sum1:', length1)

    return new_pred, new_true

# ###############################
# 保留batch_size4
#
#     global pre_bs1
#     global pre_bs2
#     global pre_bs3
#     global pre_bs4
#     global length1
#     global length2
#     global length3
#     global length4
#     y_true1 = y_true[0].clone()
#     y_true2 = y_true[1:2].clone()
#     y_true2 = torch.squeeze(y_true2, dim=0)
#     y_true3 = y_true[2:3].clone()
#     y_true3 = torch.squeeze(y_true3, dim=0)
#     y_true4 = y_true[3:].clone()
#     y_true4 = torch.squeeze(y_true4, dim=0)
#     y_pred1 = y_predict[0].clone()
#     y_pred2 = y_predict[1:2].clone()
#     y_pred2 = torch.squeeze(y_pred2, dim=0)
#     y_pred3 = y_predict[2:3].clone()
#     y_pred3 = torch.squeeze(y_pred3, dim=0)
#     y_pred4 = y_predict[3:].clone()
#     y_pred4 = torch.squeeze(y_pred4, dim=0)
#     if y_predict.shape[0] == 4:
#         pre_bs1, length1 = torchmask(y_pred1, y_true1)
#         pre_bs2, length2 = torchmask(y_pred2, y_true2)
#         pre_bs3, length3 = torchmask(y_pred3, y_true3)
#         pre_bs4, length4 = torchmask(y_pred4, y_true4)

