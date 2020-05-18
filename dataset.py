from PIL import Image
import torchvision.transforms as transforms
import torch
from torch.utils.data import Dataset, DataLoader

class EyeDataset(Dataset):
    def __init__(self, path_list, transform):
        self.transform = transform
        self.imgs = []

        for path in path_list:
            if (path.find("n_") != -1):
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


def get_train_transform():
    transform_train = transforms.Compose([
        # transforms.RandomCrop(128, padding=4),
        transforms.Resize(size=(224, 224), interpolation=2),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(180),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    return transform_train


def get_test_transform():
    transform_test = transforms.Compose([
        transforms.Resize(size=(224, 224), interpolation=2),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    return transform_test


def get_loader(img_list, transfom):
    dataset = EyeDataset(img_list, transfom)
    loader = torch.utils.data.DataLoader(dataset, batch_size = 32, shuffle=True, num_workers=2)
    return loader


def get_cross_valid_img_list(idx, valid_num, img_list):
    train_imgs = img_list[0 : idx * valid_num] + img_list[(idx+1) * valid_num:]
    valid_imgs = img_list[idx * valid_num : (idx+1) * valid_num]
    return train_imgs, valid_imgs
