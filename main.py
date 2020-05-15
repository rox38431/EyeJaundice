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
from utils import plot_figure, prepare_dir, get_last_conv_name, load_parameter, store_parameter
from dataset import get_train_transform, get_test_transform, get_loader, get_cross_valid_img_list
from dataset import EyeDataset
# from utils import progress_bar

# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
torch.cuda.set_device(0)


def train(net, optimizer, criterion, epoch, trainloader):
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = Variable(inputs).cuda(), Variable(targets).cuda()
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    print(f"Loss: {train_loss / len(trainloader):.2f} | Acc: {100.*correct/total:.2f}")
    return 100.*correct / total, train_loss / len(trainloader)


def valid(net, criterion, epoch, validloader):
    net.eval()
    valid_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(validloader):
            inputs, targets = Variable(inputs).cuda(), Variable(targets).cuda()
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            valid_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    print(f"Loss: {valid_loss / len(validloader):.2f} | Acc: {100.*correct/total:.2f}")
    return 100.*correct / total, valid_loss / len(validloader)


def test(net, criterion, epoch, testloader):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = Variable(inputs).cuda(), Variable(targets).cuda()
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    print(f"Loss: {test_loss / len(testloader):.2f} | Acc: {100.*correct/total:.2f}")
    return 100.*correct / total, test_loss / len(testloader)


def k_fold_cross_validation(net, optimizer, criterion, train_img_list, k, present_time):
    transform_train, transform_test = get_train_transform(), get_test_transform()
    valid_img_num = math.ceil(len(train_img_list) / k)
    total_best_valid_acc, total_best_valid_loss = 0, 0

    for val_idx in range(k):  # Which part of training set should be validation set
        net.load_state_dict(torch.load(f"D:\\Pro\\EyeJaundice\\weights\\{present_time}\\init_weight.pth"))
        train_imgs, valid_imgs = get_cross_valid_img_list(val_idx, valid_img_num, train_img_list)
        train_loader = get_loader(train_imgs, transform_train)
        valid_loader = get_loader(valid_imgs, transform_test)
        train_acc_list, train_loss_list, valid_acc_list, valid_loss_list = list(), list(), list(), list()
        best_valid_loss, best_valid_epoch, non_improve_count = 10000, 0, 0
       
        for epoch in range(200):
            print(f"\n({val_idx+1})Epoch: {epoch + 1}")
            train_acc, train_loss = train(net, optimizer, criterion, epoch, train_loader)
            valid_acc, valid_loss = test(net, criterion, epoch, valid_loader)

            # 以下的 list 是為了畫圖用
            train_acc_list.append(train_acc)
            train_loss_list.append(train_loss)
            valid_acc_list.append(valid_acc)
            valid_loss_list.append(valid_loss)

            if (valid_loss < best_valid_loss):
                best_valid_loss = valid_loss
                best_valid_epoch = epoch
                non_improve_count = 0
            else:
                non_improve_count += 1

            if (non_improve_count >= 10):
                break

        total_best_valid_acc += valid_acc_list[best_valid_epoch]
        total_best_valid_loss += valid_loss_list[best_valid_epoch]
        plot_figure(train_acc_list, valid_acc_list, train_loss_list, valid_loss_list, val_idx, present_time)

    print("\n----------")
    print(f"valid acc: {total_best_valid_acc / 5:.2f}, valid loss: {total_best_valid_loss / 5:.2f}")


def final_training(net, optimizer, criterion, train_img_list, present_time):
    transform_train = get_train_transform()
    trainset = EyeDataset(train_img_list, transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)

    for epoch in range(200):
        print(f"\n(train)Epoch: {epoch}")
        train(net, optimizer, criterion, epoch, train_loader)

    torch.save(net.state_dict(), f"D:\\Pro\\EyeJaundice\\weights\\{present_time}\\final_testing_weight.pth")

    return net


def final_testing(net, criterion, test_img_list):
    transform_test = get_test_transform()
    testset = EyeDataset(test_img_list, transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size = 32, shuffle=False, num_workers=2)
    test(net, criterion, 0, test_loader)


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
    
    train_img_list = glob.glob("D:\\Pro\\eye_data\\dataset\\train\\*.jpg")
    test_img_list = glob.glob("D:\\Pro\\eye_data\\dataset\\test\\*.jpg")
    random.shuffle(train_img_list)

    torch.save(net.state_dict(), f"D:\\Pro\\EyeJaundice\\weights\\{present_time}\\init_weight.pth")  # 先儲存初始的 weight, 5-fold cross validation 每次都需要先 load 初始 weigth
    total_best_valid_acc, total_best_valid_loss = 0, 0  # 用來算在 5-fold cross validation 上 accuracy 和 loss 的表現


    # Training: use 5-fold cross validation to test the generalizability
    k = 5
    k_fold_cross_validation(net, optimizer, criterion, train_img_list, k, present_time)


    # Training: use the entire training set to train the model
    net.load_state_dict(torch.load(f"D:\\Pro\\EyeJaundice\\weights\\{present_time}\\init_weight.pth"))
    random.shuffle(train_img_list)
    net = final_training(net, optimizer, criterion, train_img_list, present_time)


    # Testing: Evaluate the trained model on the test set
    print("\n========= result on testing set =========\n")
    print("testing set length info:", len(test_img_list))
    final_testing(net, criterion, test_img_list)



if __name__ == "__main__":
    main()