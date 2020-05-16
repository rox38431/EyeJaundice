import math
import torch
from torch.autograd import Variable

from dataset import get_train_transform, get_test_transform, get_loader, get_cross_valid_img_list
from dataset import EyeDataset
from utils import plot_figure


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