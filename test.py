from __future__ import print_function, division

import argparse
import os
import sys
import time

import cv2
import numpy as np
from PIL import Image
import glob

import torch.nn
import torchvision
import matplotlib.pyplot as plt
import natsort
from torch.utils.data.sampler import SubsetRandomSampler
from Data_Loader import Images_Dataset, Images_Dataset_folder
import torchsummary

import shutil
import random
from Models import Unet_dict, NestedUNet, U_Net, R2U_Net, AttU_Net, R2AttU_Net
from losses import calc_loss, mse_loss, dice_loss, masktransform, tensor_mask, L1_loss, SSIM_loss, mse_5backloss
from Metrics import dice_coeff, accuracy_score, rmse, mse, mae


# ##################################################### 尝试加入超参

def parse_args():
    '''PARAMETERS（参数）'''
    parser = argparse.ArgumentParser('training')
    # parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')  # 指定运算为cpu
    # parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')  # 指定运算为GPU
    parser.add_argument('--batch_size', type=int, default=2, help='batch size in training')  # 定义batch_size数，每次迭代使用的样本数

    parser.add_argument('--epoch', default=50, type=int, help='number of epoch in training')  # 样本的总训练次数
    parser.add_argument('--lr', default=0.0003, type=float, help='learning rate in training')  # 优化器对应学习速率
    parser.add_argument('--patience', default=10, type=int, help='lr decrease patience in training')
    parser.add_argument('--model', default=0, type=int, help='model name index[default: U_Net]')  # 选择模型训练方法
    # model_Inputs = [U_Net, R2U_Net, AttU_Net, R2AttU_Net, NestedUNet]
    parser.add_argument('--loss', default=6, type=int, help='loss option name index [default: mse]')  # loss函数选择
    # loss_input = [calc_loss, mse_loss, bce_loss, L1_loss, SSIM_loss, ssim_mse, mse_5backloss]
    parser.add_argument('--error', default=1, type=int, help='metrics error option name index [default: mse]')  # 模型评估
    # met_option = [dice_coeff, mse, rmse, mae]
    parser.add_argument('--depth', default='Float', type=str, help='depth date type [default: Float]')

    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')  # 选择adam优化器
    parser.add_argument('--save_folder', type=str, default='./model', help='experiment root')  # 指明路径
    # parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')    # 学习率衰减指数
    # png 深度图
    # parser.add_argument('--t_data', type=str, default='./data/deep0/train/5back/', help='train data images')
    # parser.add_argument('--l_data', type=str, default='./data/deep0/train/masks/', help='train data labels')
    # parser.add_argument('--test_image', type=str, default='./data/deep0/000611.png', help='in train test images ')
    # parser.add_argument('--test_label', type=str, default='./data/deep0/000611mask.png', help='in train test labels')
    # parser.add_argument('--test_folderP', type=str, default='./data/deep0/test/5back/*', help='test data images')
    # parser.add_argument('--test_folderL', type=str, default='./data/deep0/test/masks/*', help='test data images')
    # tiff 浮点深度图
    parser.add_argument('--t_data', type=str, default='./data/tiff5back/train/imgs/', help='train data images')
    parser.add_argument('--l_data', type=str, default='./data/tiff5back/train/masks/', help='train data labels')
    parser.add_argument('--test_image', type=str, default='./data/tiff5back/000611.png', help='in train test images ')
    parser.add_argument('--test_label', type=str, default='./data/tiff5back/000611.tiff', help='in train test labels')
    parser.add_argument('--test_folderP', type=str, default='./data/tiff5back/test/imgs/*', help='test data images')
    parser.add_argument('--test_folderL', type=str, default='./data/tiff5back/test/masks/*', help='test data images')
    return parser.parse_args()

    # # t_data = './data/tiff5back/train/imgs/'
    # # l_data = './data/tiff5back/train/masks/'
    # # # l_data = './data/tiff5back/train/Z_train.npy'
    # # test_image = './data/tiff5back/000611.png'
    # # test_label = './data/tiff5back/000611.tiff'
    # # test_folderP = './data/tiff5back/test/imgs/*'
    # # test_folderL = './data/tiff5back/test/masks/*'


def testmain(args):
    # args = parse_args()
    #######################################################
    # Checking if GPU is used
    #######################################################

    train_on_gpu = torch.cuda.is_available()
    device = torch.device("cuda" if train_on_gpu else "cpu")
    if not train_on_gpu:
        print('CUDA is not available. Training on CPU')
    else:
        print('CUDA is available. Training on GPU')



    #######################################################
    # Setting the basic paramters of the model
    #######################################################

    batch_size = args.batch_size
    print('batch_size = ' + str(batch_size))

    valid_size = 0.15

    epoch = args.epoch
    print('epoch = ' + str(epoch))
    print('lr= ' + str(args.lr))
    print('patience= ' + str(args.patience))
    print('loss option = ' + str(args.loss))
    print('error option = ' + str(args.error))
    print('depth date type = ' + str(args.depth))

    random_seed = random.randint(1, 100)
    print('random_seed = ' + str(random_seed))

    shuffle = True
    valid_loss_min = np.Inf
    num_workers = 0
    lossT = []
    lossL = []
    lossL.append(np.inf)
    lossT.append(np.inf)
    epoch_valid = epoch-2
    n_iter = 1
    i_valid = 0

    pin_memory = False
    if train_on_gpu:
        pin_memory = True

    # plotter = VisdomLinePlotter(env_name='Tutorial Plots')

    #######################################################
    # Setting up the model
    #######################################################

    model_Inputs = [U_Net, R2U_Net, AttU_Net, R2AttU_Net, NestedUNet]
    # passsing this string so that if it's AttU_Net or R2ATTU_Net it doesn't throw an error at torchSummary

    # model_test = model_unet(model_Inputs[0], 1, 1)  # 同步修改 》》》
    model_test = model_unet(model_Inputs[args.model], 1, 1)

    model_test.to(device)

    #######################################################
    # Getting the Summary of Model
    #######################################################
    # input_size 修改 》》》 原(3, 128, 128)
    torchsummary.summary(model_test, input_size=(1, 640, 480))

    #######################################################
    # Passing the Dataset of Images and Labels
    #######################################################

    # t_data = args.t_data
    # l_data = args.l_data
    # test_image = args.test_image
    # test_label = args.test_label
    test_folderP = args.test_folderP
    test_folderL = args.test_folderL

    #######################################################
    # Giving a transformation for input data
    #######################################################

    data_transform = torchvision.transforms.Compose([
        #  torchvision.transforms.Resize((128,128)),
        #   torchvision.transforms.CenterCrop(96),
        torchvision.transforms.ToTensor(),  # 转为张量，复合要求则会进行归一化
        # torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        torchvision.transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # #########################################################《《《《TEST《《《《《《《《《《《《《begin
    # Loading the model
    #######################################################
    New_folder = args.save_folder
    test1 = model_test.load_state_dict(torch.load('{}/Unet_D_'.format(New_folder) +
                       str(epoch) + '_' + str(batch_size)+ '/Unet_epoch_' + str(epoch)
                       + '_batchsize_' + str(batch_size) + '.pth'))

    #######################################################
    # checking if cuda is available
    #######################################################

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    #######################################################
    # Loading the model
    #######################################################

    model_test.load_state_dict(torch.load('{}/Unet_D_'.format(New_folder) +
                       str(epoch) + '_' + str(batch_size)+ '/Unet_epoch_' + str(epoch)
                       + '_batchsize_' + str(batch_size) + '.pth'))

    model_test.eval()

    #######################################################
    # opening the test folder and creating a folder for generated images
    #######################################################

    read_test_folder = glob.glob(test_folderP)
    x_sort_test = natsort.natsorted(read_test_folder)  # To sort

    read_test_folder112 = '{}/gen_images'.format(New_folder)

    if os.path.exists(read_test_folder112) and os.path.isdir(read_test_folder112):
        shutil.rmtree(read_test_folder112)

    try:
        os.mkdir(read_test_folder112)
    except OSError:
        print("Creation of the testing directory %s failed" % read_test_folder112)
    else:
        print("Successfully created the testing directory %s " % read_test_folder112)

    # For Prediction Threshold

    read_test_folder_P_Thres = '{}/pred_threshold'.format(New_folder)

    if os.path.exists(read_test_folder_P_Thres) and os.path.isdir(read_test_folder_P_Thres):
        shutil.rmtree(read_test_folder_P_Thres)

    try:
        os.mkdir(read_test_folder_P_Thres)
    except OSError:
        print("Creation of the testing directory %s failed" % read_test_folder_P_Thres)
    else:
        print("Successfully created the testing directory %s " % read_test_folder_P_Thres)

    # For Label Threshold

    read_test_folder_L_Thres = '{}/label_threshold'.format(New_folder)

    if os.path.exists(read_test_folder_L_Thres) and os.path.isdir(read_test_folder_L_Thres):
        shutil.rmtree(read_test_folder_L_Thres)

    try:
        os.mkdir(read_test_folder_L_Thres)
    except OSError:
        print("Creation of the testing directory %s failed" % read_test_folder_L_Thres)
    else:
        print("Successfully created the testing directory %s " % read_test_folder_L_Thres)

    #######################################################
    # saving the images in the files
    #######################################################

    img_test_no = 0

    for i in range(len(read_test_folder)):
        im = Image.open(x_sort_test[i])

        s = data_transform(im)
        # s.shape=[1,480,640],tensor格式
        pred = model_test(s.unsqueeze(0).cuda()).cpu()
        # s.unsqueeze(0)升维度，在原维度前[0]重新添加一个新维度,pred.shape=[1,1,480,640]
        # pred = torch.sigmoid(pred)   # sigmoid 限定输出0-1 是否应该限定为-1 到 1
        # pred = torch.tanh(pred)
        pred = pred.detach().numpy()
        # pred[0][0] = (pred[0][0] + 1) * 127.5   # ((img * 0.5) + 0.5 )* 255 反归一化
        # pred[0][0] = ((pred[0][0] * 0.5) + 0.5) * 255
        # pred[0][0] = ((pred[0][0]-pred[0][0].min())/(pred[0][0].max()-pred[0][0].min())) * 255  # 线性映射[0-1]
        pred[0][0] = (pred[0][0] * 249.756889) + 22.294105      # 采用数据集min，max的反线性变化

        # pred = threshold_predictions_p(pred) # Value kept 0.01 as max is 1 and noise is very small.

        if i % 24 == 0:
            img_test_no = img_test_no + 1

        # x1 = plt.imsave('{}/gen_images/im_epoch_'.format(New_folder) + str(epoch) + 'int_' + str(i)
        #                 + '_img_no_' + str(img_test_no) + '.png', pred[0][0])
        # plt.imsave 保存彩色RGB(A)，单通道根据CMAP进行映射。因此可以直接将[-1,1]直接保存成4通道
        # pred = np.asarray(pred[0][0][:], np.float32)
        # x1 = cv2.imwrite('{}/gen_images/im_epoch_'.format(New_folder) + str(epoch) + 'int_' + str(i)
        #                 + '_img_no_' + str(img_test_no) + '.png', pred[0][0], (cv2.IMWRITE_PNG_COMPRESSION, 9))
        cv2.imwrite('{}/gen_images/im_epoch_'.format(New_folder) + str(epoch) + 'int_' + str(i)
                    + '_img_no_' + str(img_test_no) + '.tiff', pred[0][0], (int(cv2.IMWRITE_TIFF_COMPRESSION), 1))

    ####################################################
    # Calculating the Dice Score
    ####################################################

    data_transformP = torchvision.transforms.Compose([
        # torchvision.transforms.Resize((128,128)),
        # torchvision.transforms.CenterCrop(96),
        torchvision.transforms.Grayscale(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.5], std=[0.5])
            ])

    data_transformL = torchvision.transforms.Compose([
        # torchvision.transforms.Resize((128,128)),
        # torchvision.transforms.CenterCrop(96),
        # torchvision.transforms.Grayscale(),
        # torchvision.transforms.Normalize(mean=[0.5], std=[0.5])
            ])

    #  glob.glob 返回所有匹配的文件路径列表
    read_test_folderP = glob.glob('{}/gen_images/*'.format(New_folder))
    x_sort_testP = natsort.natsorted(read_test_folderP)


    read_test_folderL = glob.glob(test_folderL)
    x_sort_testL = natsort.natsorted(read_test_folderL)  # To sort


    dice_score123 = 0.0
    x_count = 0
    x_dice = 0

    for i in range(len(read_test_folderP)):

        # x = Image.open(x_sort_testP[i])   # tiff
        x = cv2.imread(x_sort_testP[i], -1)  # cv2读取tiff，数据为numpy.ndarray
        # x = Image.fromarray(x)
        # s = data_transformP(x)
        # s = torch.transpose(torch.transpose(s, 0, 1), 1, 2)
        # s = s.cpu().detach().numpy()
        s = np.array(x, np.float32)
        # s = threshold_predictions_v(s)
        # save the images 2/255 ：threshold_predictions_v（）

        # y = Image.open(x_sort_testL[i])
        y = cv2.imread(x_sort_testL[i], -1)  # cv2读取tiff，数据为numpy.ndarray
        # y_norm = (y-y.min())/(y.max()-y.min())    # 深度标准化，最小值为0，最大值为1：(l-min)/(max-min)
        # y = Image.fromarray(y)
        # s2 = data_transformL(y)
        s3 = np.array(y, np.float32)
        # s3 = s3.reshape(s3.shape[0], s3.shape[1], 1)
        # s2 = threshold_predictions_v(s2)

        #############
        # 去除back——》pred_threshold
        ss = masktransform(s, s3)    # s(480,640,1) s3(480,640)

        # save the Images test_label
        y1 = cv2.imwrite('{}/label_threshold/im_epoch_'.format(New_folder) + str(epoch) + 'int_' + str(i)
                        + '_img_no_' + str(img_test_no) + '.png', s3, (cv2.IMWRITE_PNG_COMPRESSION, 9))

        # x1 = plt.imsave('{}/pred_threshold/im_epoch_'.format(New_folder) + str(epoch) + 'int_' + str(i)
        #                 + '_img_no_' + str(img_test_no) + '.png', s)
        x1 = cv2.imwrite('{}/pred_threshold/im_epoch_'.format(New_folder) + str(epoch) + 'int_' + str(i)
                         + '_img_no_' + str(img_test_no) + '.png', ss, (cv2.IMWRITE_PNG_COMPRESSION, 9))
        # cv2.imwrite('{}/pred_threshold/im_epoch_'.format(New_folder) + str(epoch) + 'int_' + str(i)
        #            + '_img_no_' + str(img_test_no) + '.tiff', ss, (int(cv2.IMWRITE_TIFF_COMPRESSION), 1))

        #############
        # 对images、label进行norm后求error
        # images norm[0.5,0.5]
        ss_norm = Image.fromarray(ss)
        ss_norm = data_transformP(ss_norm)
        ss_norm = torch.transpose(torch.transpose(ss_norm, 0, 1), 1, 2)
        ss_norm = ss_norm.cpu().detach().numpy()
        ss_norm = np.array(ss_norm, np.float32)     # [-1,1]
        ss_norm = (ss_norm+1)/2     # [0,1]

        # label norm:
        y_norm = (y-y.min())/(y.max()-y.min())  # 深度标准化，最小值为0，最大值为1：(l-min)/(max-min)
        y_norm = Image.fromarray(y_norm)
        y_norm = data_transformL(y_norm)
        y_norm = np.array(y_norm, np.float32)
        y_norm = y_norm.reshape(y_norm.shape[0], y_norm.shape[1], 1)

        # 模型评估： ss_norm:[-1,1] ; y_norm:[0,1]
        # total = dice_coeff(s, s3)  # total全为1？？
        # total = rmse(ss, s3)
        met_option = [dice_coeff, mse, rmse, mae]
        total = error_metrics(met_option[args.error], ss_norm, y_norm)
        print(total)

        if total <= 0.3:
            x_count += 1
        if total > 0.3:
            x_dice = x_dice + total
        dice_score123 = dice_score123 + total

        ################
        # 还原图片

        # ss = (ss+1)*127.5
        # s3 = s3 * (y.max()-y.min()) + y.min()
    print('Dice Score : ' + str(dice_score123/len(read_test_folderP)))

    # print(x_count)
    # print(x_dice)
    # print('Dice Score : ' + str(float(x_dice/(len(read_test_folderP)-x_count))))


# main中提取
def model_unet(model_input, in_channel=1, out_channel=1):  # 测试修改成3->1 》》》
    model_test = model_input(in_channel, out_channel)
    return model_test


def error_metrics(met_option, y_true, y_predict):  # 测试修改成3->1 》》》
    error = met_option(y_true, y_predict)
    return error


# 控制台输出记录到文件
class Logger(object):
    def __init__(self, file_name="Default.log", stream=sys.stdout):
        self.terminal = stream
        self.log = open(file_name, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


if __name__ == '__main__':
    args = parse_args()

    # 自定义目录存放日志文件
    log_path = '{}/Logs/'.format(args.save_folder)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    # 日志文件名按照程序运行时间设置
    log_file_name = log_path + 'testlog-' +  'epoch{}-loss{}-er{}'.format(args.epoch, args.loss, args.error)  + '.log'
    # 记录正常的 print 信息
    sys.stdout = Logger(log_file_name)
    # 记录 traceback 异常信息
    sys.stderr = Logger(log_file_name)

    testmain(args)
