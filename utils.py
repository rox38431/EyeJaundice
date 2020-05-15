import os
import matplotlib.pyplot as plt
import torch


def get_last_conv_name(net):
    layer_name = None
    for name, m in net.named_modules():
        if isinstance(m, nn.Conv2d):
            layer_name = name
    return layer_name


def prepare_dir(present_time):
    if (os.path.exists("D:\\Pro\\EyeJaundice\\weights") == False):
        os.mkdir("D:\\Pro\\EyeJaundice\\weights")

    if (os.path.exists("D:\\Pro\\EyeJaundice\\plt_figure") == False):
        os.mkdir("D:\\Pro\\EyeJaundice\\plt_figure")

    if (os.path.exists(f"D:\\Pro\\EyeJaundice\\weights\\{present_time}") == False):
        os.mkdir(f"D:\\Pro\\EyeJaundice\\weights\\{present_time}")

    if (os.path.exists(f"D:\\Pro\\EyeJaundice\\plt_figure\\{present_time}") == False):
        os.mkdir(f"D:\\Pro\\EyeJaundice\\plt_figure\\{present_time}")


def plot_figure(train_acc_list, valid_acc_list, train_loss_list, valid_loss_list, val_idx, present_time):
    plt.plot(train_acc_list)
    plt.plot(valid_acc_list)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.savefig(f"D:\\Pro\\EyeJaundice\\plt_figure\\{present_time}\\train_acc_{val_idx + 1}.png")
    plt.close()

    plt.plot(train_loss_list)
    plt.plot(valid_loss_list)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(f"D:\\Pro\\EyeJaundice\\plt_figure\\{present_time}\\train_loss_{val_idx + 1}.png")
    plt.close()


def load_parameter(net, date):
    assert os.path.isdir(f"D:\\Pro\\EyeJaundice\\weights\\{date}\\checkpoint.pth"), 'Error: no checkpoint directory found!'

    checkpoint = torch.load(f"D:\\Pro\\EyeJaundice\\weights\\{date}\\checkpoint.pth")
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['best_acc']
    start_epoch = checkpoint['epoch']
    
    return net, best_acc, start_epoch


def store_parameter(epoch, net, optimizer, best_acc, best_loss, present_time):
    present_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    os.mkdir(f"D:\\Pro\\EyeJaundice\\weights\\{present_time}")

    torch.save({
        "epoch": epoch + 1,
        "model_weight": net.state_dict(),
        "optim_weight": optimizer.state_dict(),
        "best_acc": best_acc,
        "best_loss": best_loss
    }, f"D:\\Pro\\EyeJaundice\\weights\\{present_time}\\checkpoint.pth")