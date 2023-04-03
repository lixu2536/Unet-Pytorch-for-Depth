from __future__ import print_function, division

import argparse
import os
import sys

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
# 表示当前可以被pytorch环境检测到的显卡序号
import cv2
import numpy as np
from PIL import Image
import glob

# import SimpleITK as sitk
from torch import optim
import torch.utils.data
import torch
import torch.nn.functional as F

import torch.nn
import torchvision
import matplotlib.pyplot as plt
import natsort
from torch.utils.data.sampler import SubsetRandomSampler
from Data_Loader import Images_Dataset, Images_Dataset_folder
import torchsummary
# from torch.utils.tensorboard import SummaryWriter
# from tensorboardX import SummaryWriter

import shutil
import random
from Models import Unet_dict, NestedUNet, U_Net, R2U_Net, AttU_Net, R2AttU_Net
from losses import calc_loss, mse_loss, dice_loss, bce_loss, L1_loss, SSIM_loss, ssim_mse, mse_5backloss, ssim_5back
from ploting import plot_kernels, LayerActivations, input_images, plot_grad_flow
from Metrics import dice_coeff, accuracy_score, rmse, mse, mae
import time
import pandas
# from ploting import VisdomLinePlotter
# from visdom import Visdom


# #####################################################

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
    # loss_input = [calc_loss, mse_loss, bce_loss, L1_loss, SSIM_loss, ssim_mse, mse_5backloss, ssim_5back]
    parser.add_argument('--error', default=1, type=int, help='metrics error option name index [default: mse]')  # 模型评估
    # met_option = [dice_coeff, mse, rmse, mae]
    parser.add_argument('--depth', default='Float', type=str, help='depth date type [default: Float]')

    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')  # 选择adam优化器
    parser.add_argument('--save_folder', type=str, default='./model', help='experiment root')  # 指明路径
    # parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')    # 学习率衰减指数
    # # png 深度图
    # parser.add_argument('--t_data', type=str, default='./data/deep0/train/5back/', help='train data images')
    # parser.add_argument('--l_data', type=str, default='./data/deep0/train/masks/', help='train data labels')
    # parser.add_argument('--test_image', type=str, default='./data/deep0/000611.png', help='in train test images ')
    # parser.add_argument('--test_label', type=str, default='./data/deep0/000611mask.png', help='in train test labels')
    # parser.add_argument('--test_folderP', type=str, default='./data/deep0/test/5back/*', help='test data images')
    # parser.add_argument('--test_folderL', type=str, default='./data/deep0/test/masks/*', help='test data images')
    # # tiff 浮点深度图
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


def main(args):
    # args = parse_args()
    #######################################################
    # Checking if GPU is used
    #######################################################

    train_on_gpu = torch.cuda.is_available()
    device = torch.device("cuda" if train_on_gpu else "cpu")
    if not train_on_gpu:
        print('CUDA is not available. Training on CPU')
    else:
        print('CUDA is available. Training on GPU:{}-{}'.format(torch.cuda.device_count(),
                                                                torch.cuda.get_device_name()))

    # device = torch.device("cuda" if train_on_gpu else "cpu")

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
    valid_loss_min = np.Inf  # 导致loss Train/Validation Loss: nan
    # valid_loss_min = 500
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

    model_test = model_test.to(device)    # 原
    # if torch.cuda.device_count()>1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    #     model_test = torch.nn.DataParallel(model_test, device_ids=[0, 1], output_device=[1])
    #
    # model_test.to(device)

    #######################################################
    # Getting the Summary of Model
    #######################################################
    # input_size 修改 》》》 原(3, 128, 128)
    torchsummary.summary(model_test, input_size=(1, 640, 480))

    #######################################################
    # Passing the Dataset of Images and Labels
    #######################################################

    t_data = args.t_data
    l_data = args.l_data
    test_image = args.test_image
    test_label = args.test_label
    # test_folderP = args.test_folderP
    # test_folderL = args.test_folderL

    Training_Data = Images_Dataset_folder(t_data,
                                          l_data)
    # 数据加载为dataset格式
    # 标签为单通道深度图
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

    #######################################################
    # Trainging Validation Split
    #######################################################

    num_train = len(Training_Data)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]     # 重新注意采样策略
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    # 训练集 train_loss
    train_loader = torch.utils.data.DataLoader(Training_Data, batch_size=batch_size, sampler=train_sampler,
                                               num_workers=num_workers, pin_memory=pin_memory)
    # 测试集 val_loss
    valid_loader = torch.utils.data.DataLoader(Training_Data, batch_size=batch_size, sampler=valid_sampler,
                                               num_workers=num_workers, pin_memory=pin_memory)

    #######################################################
    # Using Adam as Optimizer
    #######################################################

    initial_lr = args.lr
    opt = torch.optim.Adam(model_test.parameters(), lr=initial_lr, betas=(0.5, 0.999) , weight_decay=1e-4 )
    # opt = optim.SGD(model_test.parameters(), lr = initial_lr, momentum=0.99)  ## l2正则化参数

    # MAX_STEP = int(epoch)   # 定义余弦变化从最高到最低的间隔
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_STEP, eta_min=1e-8, verbose=True)  # 余弦退火调度
    # scheduler = optim.lr_scheduler.CosineAnnealingLr(opt, epoch, 1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=args.patience,
                                                           verbose=True, min_lr=0, eps=1e-8)
    #  新定义学习率调整，指标不变化时，进行调整

    #######################################################
    # Writing the params to tensorboard
    #######################################################

    # writer1 = SummaryWriter()
    # dummy_inp = torch.randn(1, 3, 128, 128)
    # model_test.to('cpu')
    # writer1.add_graph(model_test, model_test(torch.randn(3, 3, 128, 128, requires_grad=True)))
    # model_test.to(device)

    #######################################################
    # Creating a Folder for every data of the program
    #######################################################

    New_folder = args.save_folder   # './model'

    if os.path.exists(New_folder) and os.path.isdir(New_folder):
        shutil.rmtree(New_folder)

    try:
        os.mkdir(New_folder)
    except OSError:
        print("Creation of the main directory '%s' failed " % New_folder)
    else:
        print("Successfully created the main directory '%s' " % New_folder)

    #######################################################
    # Setting the folder of saving the predictions
    #######################################################

    read_pred = '{}/pred'.format(New_folder)

    #######################################################
    # Checking if prediction folder exixts
    #######################################################

    if os.path.exists(read_pred) and os.path.isdir(read_pred):
        shutil.rmtree(read_pred)

    try:
        os.mkdir(read_pred)
    except OSError:
        print("Creation of the prediction directory '%s' failed of loss" % read_pred)
    else:
        print("Successfully created the prediction directory '%s' of loss" % read_pred)

    #######################################################
    # checking if the model exists and if true then delete
    #######################################################

    read_model_path = '{}/Unet_D_'.format(New_folder) + str(epoch) + '_' + str(batch_size)

    if os.path.exists(read_model_path) and os.path.isdir(read_model_path):
        shutil.rmtree(read_model_path)
        print('Model folder there, so deleted for newer one')

    try:
        os.mkdir(read_model_path)
    except OSError:
        print("Creation of the model directory '%s' failed" % read_model_path)
    else:
        print("Successfully created the model directory '%s' " % read_model_path)

    #######################################################
    # Training loop
    #######################################################
    progress_train = []
    progress_val = []
    for i in range(epoch):

        train_loss = 0.0
        valid_loss = 0.0
        since = time.time()

        #######################################################
        # Training Data
        #######################################################

        model_test.train()
        k = 1
        # scheduler.step(i)
        # scheduler.get_last_lr()
        # scheduler.step(valid_loss)    # 监控指标，此处第7次改变，以后每5次降一次

        for x, y in train_loader:   # 跳入train_loader的getitem
            x, y = x.to(device), y.to(device)

            # If want to get the input images with their Augmentation - To check the data flowing in net
            # input_images(x, y, i, n_iter, k)  # 若符合batch=1，注释

            # grid_img = torchvision.utils.make_grid(x)
            # writer1.add_image('images', grid_img, 0)

            # grid_lab = torchvision.utils.make_grid(y)
            # opt.zero_grad()

            y_pred = model_test(x)      # 更改tiff 后，出现显存不足
            # lossT = calc_loss(y_pred, y)     # Dice_loss Used
            ################
            # # loss 前进行背景去除，加上后运行速度很慢。。。
            # if y_pred.shape[0] == 2:
            #     pre_bs1 = tensor_mask(y_pred[0, :, :, :], y[0, :, :, :])
            #     pre_bs2 = tensor_mask(y_pred[1, :, :, :], y[1, :, :, :])
            #     y_pred = torch.stack((pre_bs1, pre_bs2), dim=0).float()
            # else:
            #     y_pred = torch.unsqueeze(tensor_mask(y_pred[-3], y[-3]), dim=0)

            loss_input = [calc_loss, mse_loss, bce_loss, L1_loss, SSIM_loss, ssim_mse, mse_5backloss, ssim_5back]
            # lossT = mse_loss(y_pred, y)
            lossT = loss_option(loss_input[args.loss], y_pred, y)

            train_loss = train_loss + lossT.item() * x.size(0)  # size(0) = batch_size  ######
            opt.zero_grad()
            # lossT.backward(retain_graph=True)     # 多个backward使用，保留梯度
            lossT.backward()
            # plot_grad_flow(model_test.named_parameters(), n_iter)
            opt.step()  # 原优化参数更新
            # x_size = lossT.item() * x.size(0)
            k = 2
        # opt.step()
        # 优化参数更新 为解决inplace operation: [torch.cuda.FloatTensor [1, 32, 1, 1]] is at version 260，
        # 运行正常，但初始loss=400、120、13、0.3，推测此处更新问题导致结果异常

        #######################################################
        # Validation Step
        #######################################################

        model_test.eval()
        torch.no_grad()  # to increase the validation process uses less memory

        for x1, y1 in valid_loader:
            x1, y1 = x1.to(device), y1.to(device)

            y_pred1 = model_test(x1)

            # # lossL = calc_loss(y_pred1, y1)     # Dice_loss Used
            # lossL = mse_loss(y_pred1, y1)

            loss_input = [calc_loss, mse_loss, bce_loss, L1_loss, SSIM_loss, ssim_mse, mse_5backloss, ssim_5back]
            lossL = loss_option(loss_input[args.loss], y_pred1, y1)

            valid_loss += lossL.item() * x1.size(0)
            x_size1 = lossL.item() * x1.size(0)

        #######################################################
        # Saving the predictions in train loop
        #######################################################

        im_tb = Image.open(test_image)
        # # im_label = Image.open(test_label)
        # im_label = cv2.imread(test_label, -1)  # cv2读取tiff，数据为numpy.ndarray
        # im_label = Image.fromarray(im_label)  # 将cv格式转为pil格式，用于后续的transform
        s_tb = data_transform(im_tb)
        # s_label = data_transform(im_label)
        # s_label = s_label.detach().numpy()

        pred_tb = model_test(s_tb.unsqueeze(0).to(device)).cpu()
        pred_tb = torch.sigmoid(pred_tb)    # sigmoid限制输出0-1
        # pred_tb = torch.tanh(pred_tb)
        pred_tb = pred_tb.detach().numpy()  # pred_tb = [1,1,480,640]，[0,1]，float32
        # pred_tb[0][0] = (pred_tb[0][0] + 1) * 127.5
        # pred_tb[0][0] = ((pred_tb[0][0] * 0.5) + 0.5) * 255
        # pred_tb[0][0] = pred_tb[0][0]*255

        # pred_tb = threshold_predictions_v(pred_tb)

        x1 = plt.imsave(
            '{}/pred/img_iteration_'.format(New_folder) + str(n_iter) + '_epoch_'
            + str(i) + '.png', pred_tb[0][0])
        # x1 = cv2.imwrite('./model/pred/img_iteration_' + str(n_iter) + '_epoch_'
        #     + str(i) + '.png', pred_tb[0][0], (cv2.IMWRITE_PNG_COMPRESSION, 9))

        # pred_tb[0][0]
        # accuracy = accuracy_score(pred_tb[0][0], s_label)
        # print(accuracy)

        #######################################################
        # To write in Tensorboard
        #######################################################

        train_loss = train_loss / len(train_idx)
        valid_loss = valid_loss / len(valid_idx)

        if (i+1) % 1 == 0:
            print('Epoch: {}/{} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(i + 1, epoch, train_loss,
                                                                                          valid_loss))
        progress_train.append(train_loss)
        progress_val.append(valid_loss)
        #######################################################
        # Early Stopping
        #######################################################

        if valid_loss <= valid_loss_min and epoch_valid >= i: # and i_valid <= 2:

            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model '.format(valid_loss_min, valid_loss))
            torch.save(model_test.state_dict(),'{}/Unet_D_'.format(New_folder) +
                                                  str(epoch) + '_' + str(batch_size) + '/Unet_epoch_' + str(epoch)
                                                  + '_batchsize_' + str(batch_size) + '.pth')
            # print(accuracy)
            if round(valid_loss, 4) == round(valid_loss_min, 4):
                print(i_valid)
                i_valid = i_valid+1
            valid_loss_min = valid_loss
            # if i_valid ==3:
            #   break

        scheduler.step(valid_loss)  # 监控指标 调整到此处 监测正常
        #######################################################
        # Extracting the intermediate layers
        #######################################################

        #####################################
        # for kernals
        #####################################
        x1 = torch.nn.ModuleList(model_test.children())

        #####################################
        # for images
        #####################################
        x2 = len(x1)
        dr = LayerActivations(x1[x2-1])  # Getting the last Conv Layer

        img = Image.open(test_image)
        s_tb = data_transform(img)

        pred_tb = model_test(s_tb.unsqueeze(0).to(device)).cpu()
        pred_tb = torch.sigmoid(pred_tb)
        pred_tb = pred_tb.detach().numpy()

        plot_kernels(dr.features, n_iter, 7, cmap="rainbow")
        # 输出彩色图像的map
        plt.close()

        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        n_iter += 1

    # 绘制loss图
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.plot(progress_train, color="red", label="train loss")    # x and y must have same first dimension,
    plt.plot(progress_val, color="blue", label="validation loss")  # but have shapes (1,) and (5,)
    # 设置坐标轴范围
    plt.xlim((0, args.epoch)), plt.ylim((0, 2))
    my_x_ticks = np.arange(0, args.epoch, 5)    # start, destination, step
    plt.xticks(my_x_ticks)
    plt.legend()  # 显示图例
    plt.xlabel("epoch"), plt.ylabel("loss"), plt.title("loss in training")
    plt.savefig('{}/loss_in_train.jpg'.format(args.save_folder), dpi=500, bbox_inches='tight')
    plt.show()  # 画图

    #######################################################
    # closing the tensorboard writer   训练及测试结束，下面进行验证
    #######################################################

    # writer1.close()

    #######################################################
    # if using dict
    #######################################################

    # model_test.filter_dict


# main中提取
def model_unet(model_input, in_channel=1, out_channel=1):  # 测试修改成3->1 》》》
    model_test = model_input(in_channel, out_channel)
    return model_test


def loss_option(loss_input, y_predict, y_true):
    loss = loss_input(y_predict, y_true)
    return loss


# # 控制台输出记录到文件
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

    # # 自定义目录存放日志文件
    log_path = './Logs/'
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    # 日志文件名按照程序运行时间设置
    log_file_name = log_path + 'trainlog-' + 'epoch{}-loss{}-er{}'.format(args.epoch, args.loss, args.error) + '.log'
    # 记录正常的 print 信息
    sys.stdout = Logger(log_file_name)
    # 记录 traceback 异常信息
    sys.stderr = Logger(log_file_name)

    main(args)