'''Train CIFAR10 with PyTorch.'''
import os
import argparse
import glob
import random
import math
import time
import shutil

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable

from PIL import Image
import matplotlib.pyplot as plt

from models import *
from utils import prepare_dir, get_last_conv_name, load_parameter, store_parameter
from train_test import train, test, k_fold_cross_validation, final_training, final_testing
from grad_CAM import CAM

# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
torch.cuda.set_device(1)


def main():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    present_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())

    print("Prepare directory...")
    prepare_dir(present_time)

    print("Prepare model...")
    net = EfficientNetB0().cuda()

    best_acc, start_epoch = 0, 0 
    if args.resume:
        print("Loading checkpoint...")
        net, best_acc, start_epoch = load_parameter(net, date)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)
    # optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    
    train_img_list = glob.glob("./../eye_data/dataset/train/*")
    test_img_list = glob.glob("./../eye_data/dataset/test/*")
    random.shuffle(train_img_list)

    torch.save(net.state_dict(), f"./weights/{present_time}/init_weight.pth")  # 先儲存初始的 weight, 5-fold cross validation 每次都需要先 load 初始 weigth
    total_best_valid_acc, total_best_valid_loss = 0, 0  # 用來算在 5-fold cross validation 上 accuracy 和 loss 的表現


    # Training: use 5-fold cross validation to test the generalizability
    k = 5
    k_fold_cross_validation(net, optimizer, criterion, train_img_list, k, present_time)


    # Training: use the entire training set to train the model
    net.load_state_dict(torch.load(f"./weights/{present_time}/init_weight.pth"))
    random.shuffle(train_img_list)
    net = final_training(net, optimizer, criterion, train_img_list, present_time)


    # Testing: Evaluate the trained model on the test set
    print("\n========= result on testing set =========\n")
    print("testing set length info:", len(test_img_list))
    final_testing(net, criterion, test_img_list)
    store_parameter(120, net, optimizer, -1, -1, present_time)

    CAM(net, test_img_list)

if __name__ == "__main__":
    main()
