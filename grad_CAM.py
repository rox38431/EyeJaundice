# -*- coding: utf-8 -*-

import argparse
import os
import re

import cv2
import numpy as np
import torch
from skimage import io
from torch import nn
from models import EfficientNetB0
from PIL import Image
import torchvision.transforms as transforms
from torch.autograd import Variable
import glob

from interpretability.grad_cam import GradCAM, GradCamPlusPlus
from interpretability.guided_back_propagation import GuidedBackPropagation

from models import *
from utils import load_parameter

torch.cuda.set_device(1)


def get_net(date):
    net = EfficientNetB0().cuda()
    net.load_state_dict(torch.load(f"./weights/{date}/final_testing_weight.pth"))
    return net


def get_last_conv_name(net):
    layer_name = None
    for name, m in net.named_modules():
        if isinstance(m, nn.Conv2d):
            layer_name = name
    return layer_name


def prepare_input(image):
    image = image.copy()

    means = np.array([0.4914, 0.4822, 0.4465])
    stds = np.array([0.2023, 0.1994, 0.2010])
    image -= means
    image /= stds

    image = np.ascontiguousarray(np.transpose(image, (2, 0, 1)))  # channel first
    image = image[np.newaxis, ...]  # 增加batch维

    return torch.tensor(image, requires_grad=True)


def gen_cam(image, mask):
    """
    生成CAM图
    :param image: [H,W,C],原始图像
    :param mask: [H,W],范围0~1
    :return: tuple(cam,heatmap)
    """
    # mask转为heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    heatmap = heatmap[..., ::-1]  # gbr to rgb

    # 合并heatmap到原始图像
    cam = heatmap + np.float32(image)
    return norm_image(cam), (heatmap * 255).astype(np.uint8)


def norm_image(image):
    """
    标准化图像
    :param image: [H,W,C]
    :return:
    """
    image = image.copy()
    image -= np.max(np.min(image), 0)
    image /= np.max(image)
    image *= 255.
    return np.uint8(image)


def gen_gb(grad):
    """
    生guided back propagation 输入图像的梯度
    :param grad: tensor,[3,H,W]
    :return:
    """
    # 标准化
    grad = grad.cpu().data.numpy()
    gb = np.transpose(grad, (1, 2, 0))
    return gb


def save_image(image_dicts, input_image_name, network, output_dir):
    prefix = os.path.splitext(input_image_name)[0]
    for key, image in image_dicts.items():
        io.imsave(os.path.join(output_dir, '{}-{}-{}.jpg'.format(prefix, network, key)), image)


def CAM(net, img_list, date, input_type):
    # img_list = glob.glob("./../eye_data/dataset/test/*")
    correct = 0
    for img_path in img_list:
        file_name = img_path.split("/")[-1]
        class_id = 1 if file_name[0] == 'n' else 2
        class_id -=  1 
        img = Image.open(img_path) 
        transform_test = transforms.Compose([
            transforms.Resize(size=(224, 224), interpolation=2),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        # img = transform_test(img)
        # img = img[np.newaxis, ...]  # 增加batch维 
        # img = Variable(img).cuda()
        # output = net(img)  # size(1, 2)
        
        inputs = transform_test(img)
        inputs = inputs[np.newaxis, ...]  # 增加batch维
        inputs = torch.tensor(inputs, requires_grad=True)
        inputs = Variable(inputs, requires_grad=True).cuda()
        inputs.retain_grad()
        output = net(inputs)
       
        img = io.imread(img_path)
        img = np.float32(cv2.resize(img, (224, 224))) / 255


        # print(output.data, class_id-1)

        _, predicted = output.max(1)
        # print(predicted[0].item())
        if (predicted[0].item() == class_id - 1):
            correct += 1

        image_dict = {}

        # Grad-CAM
        layer_name = get_last_conv_name(net)
        grad_cam = GradCAM(net, layer_name)
        mask = grad_cam(inputs, class_id)  # cam mask
        image_dict['cam'], image_dict['heatmap'] = gen_cam(img, mask)
        grad_cam.remove_handlers()


        # Grad-CAM++
        grad_cam_plus_plus = GradCamPlusPlus(net, layer_name)
        mask_plus_plus = grad_cam_plus_plus(inputs, class_id)  # cam mask
        image_dict['cam++'], image_dict['heatmap++'] = gen_cam(img, mask_plus_plus)
        grad_cam_plus_plus.remove_handlers()


        # GuidedBackPropagation
        gbp = GuidedBackPropagation(net)
        inputs.grad.zero_()  # 梯度置零
        grad = gbp(inputs)

        gb = gen_gb(grad)
        image_dict['gb'] = norm_image(gb)
        # 生成Guided Grad-CAM
        cam_gb = gb * mask[..., np.newaxis]
        image_dict['cam_gb'] = norm_image(cam_gb)

        save_image(image_dict, file_name, "EfficientNet-B0", f"./grad_cam_result/{input_type}_{date}")
        
    print("Test the accuracy on Grad-CAM:", correct / len(img_list))


