import numpy as np
import torch
from scipy import spatial

from sklearn import metrics # need [H,W]
# y = np.array([1,1])
# y_hat = np.array([2,3])
# MSE = metrics.mean_squared_error(y, y_hat)
# RMSE = metrics.mean_squared_error(y, y_hat)**0.5
# MAE = metrics.mean_absolute_error(y, y_hat)
# MAPE = metrics.mean_absolute_percentage_error(y, y_hat)

# MSE = np.mean(np.square(y - y_hat))
# RMSE = np.sqrt(np.mean(np.square(y - y_hat)))
# MAE = np.mean(np.abs(y-y_hat))
# MAPE = np.mean(np.abs((y - y_hat) / y)) * 100


def dice_coeff(im1, im2, empty_score=1.0):
    """Calculates the dice coefficient for the images"""

    im1 = np.asarray(im1).astype(np.bool_)
    im2 = np.asarray(im2).astype(np.bool_)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im1 = im1 > 10
    im2 = im2 > 10
    # 原来输出为0-1，根据colormap显示，现在图像乘以255

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)
    #print(im_sum)

    return 2. * intersection.sum() / im_sum


def mse(y_true, y_predict):
    return np.mean(np.square(y_true - y_predict))


def rmse(y_true, y_predict):
    return np.sqrt(np.mean(np.square(y_true - y_predict)))


def mae(y_true, y_predict):
    return np.mean(np.abs(y_true, y_predict))


def numeric_score(prediction, groundtruth):
    """Computes scores:
    FP = False Positives
    FN = False Negatives
    TP = True Positives
    TN = True Negatives
    return: FP, FN, TP, TN"""

    FP = np.float(np.sum((prediction == 1) & (groundtruth == 0)))
    FN = np.float(np.sum((prediction == 0) & (groundtruth == 1)))
    TP = np.float(np.sum((prediction == 1) & (groundtruth == 1)))
    TN = np.float(np.sum((prediction == 0) & (groundtruth == 0)))

    return FP, FN, TP, TN


def accuracy_score(prediction, groundtruth):
    """
    Getting the accuracy of the model
    混淆矩阵 验证准确率
    """

    FP, FN, TP, TN = numeric_score(prediction, groundtruth)
    N = FP + FN + TP + TN
    accuracy = np.divide(TP + TN, N)
    return accuracy * 100.0