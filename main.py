'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

import os
import argparse
import glob
import random
import math
from PIL import Image

from models import *
# from utils import progress_bar

# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
torch.cuda.set_device(2)

class EyeDataset(Dataset):
    def __init__(self, path_list, transform):
        self.transform = transform
        self.imgs = []

        for path in path_list:
            if (path.find("normal_") != -1):
                label = 0  # normal
            else:
                label = 1  # yellow

            self.imgs.append((path, label))

    def __getitem__(self, index):
        path, label = self.imgs[index]
        img = Image.open(path)
        img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.imgs)


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    # transforms.RandomCrop(128, padding=4),
    torchvision.transforms.Resize(size=128, interpolation=2),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(180),
    transforms.ToTensor(),
    transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5)),
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    torchvision.transforms.Resize(size=128, interpolation=2),
    transforms.ToTensor(),
    transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5)),
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


cross_num = 5
train_img_list = glob.glob("/tmp2/jojo/EyeJaundice/data/only_eyes/train/*/*.JPG")
random.shuffle(train_img_list)
img_num = math.ceil(len(train_img_list) / cross_num)
print(len(train_img_list), img_num)

test_img_list = glob.glob("/tmp2/jojo/EyeJaundice/data/only_eyes/test/*/*.JPG")


# Model
print('==> Building model..')
# net = VGG('VGG19')
# net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
net = EfficientNetB0()
net = net.cuda()
# if device == 'cuda':
#     net = torch.nn.DataParallel(net)
#     cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)


def train(epoch, trainloader):

    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.cuda(), targets.cuda()
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
    return 100.*correct/total, train_loss / len(trainloader)

def valid(epoch, validloader):

    net.eval()
    valid_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(validloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            valid_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    print(f"Loss: {valid_loss / len(validloader):.2f} | Acc: {100.*correct/total:.2f}")
    return 100.*correct / total, valid_loss / len(validloader)


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    print(f"Loss: {test_loss / len(testloader):.2f} | Acc: {100.*correct/total:.2f}")

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc

torch.save(net.state_dict(), "weights/init_weight.pt")
valid_acc = 0
valid_loss = 0
for validation_index in range(cross_num):  # which part of training set should be validation set
    net.load_state_dict(torch.load("weights/init_weight.pt"))
    valid_imgs = train_img_list[validation_index*50:(validation_index+1)*50]
    train_imgs = train_img_list[0:validation_index*50] + train_img_list[(validation_index+1)*50:]
    # train_imgs = train_img_list[:]
    print(f"train len: {len(train_imgs)}")

    trainset = EyeDataset(train_imgs, transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size = 32, shuffle=True, num_workers=2)

    validset = EyeDataset(valid_imgs, transform_test)
    valid_loader = torch.utils.data.DataLoader(validset, batch_size = 32, shuffle=False, num_workers=2)

    for epoch in range(200):
        print(f"\n({validation_index+1})Epoch: {epoch}")
        train(epoch, train_loader)
        acc, loss = valid(epoch, valid_loader)
        if (epoch == 199):
            valid_acc += acc
            valid_loss += loss

print(f"valid acc: {valid_acc / 5:.2f}, valid loss: {valid_loss / 5:.2f}")

net.load_state_dict(torch.load("weights/init_weight.pt"))
min_loss = 10000
non_improve_count = 0

random.shuffle(train_img_list)
training_num = int(len(train_img_list) * 0.85)

train_imgs = train_img_list[:training_num]
trainset = EyeDataset(train_imgs, transform_train)
train_loader = torch.utils.data.DataLoader(trainset, batch_size = 32, shuffle=True, num_workers=2)

valid_imgs = train_img_list[training_num:]
validset = EyeDataset(valid_imgs, transform_test)
valid_loader = torch.utils.data.DataLoader(validset, batch_size = 32, shuffle=False, num_workers=2)

print("training set length info:", len(train_img_list), len(train_imgs), len(valid_imgs))


for epoch in range(200):
    print(f"\n(train)Epoch: {epoch}")
    train(epoch, train_loader)
    valid_acc, valid_loss = valid(epoch, valid_loader)

    if (valid_loss < min_loss):
        torch.save(net.state_dict(), "weights/best_weight.pt")
        min_loss = valid_loss
        non_improve_count = 0
    else:
        non_improve_count += 1

    if (non_improve_count == 5):
        pass
        # break

print("\n========= result on testing set =========\n")
print("testing set length info:", len(test_img_list))

testset = EyeDataset(test_img_list, transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size = 32, shuffle=False, num_workers=2)
valid(0, test_loader)

