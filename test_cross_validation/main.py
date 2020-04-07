import torchvision
import torchvision.transforms as transforms
import cv2
from PIL import Image
import numpy as np

im = cv2.imread("/tmp2/jojo/EyeJaundice/data/only_eye/train/normal/normal_1.JPG")
print(type(im), im.shape)

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


img = Image.open("/tmp2/jojo/EyeJaundice/data/only_eye/train/normal/normal_1.JPG")
print(img.size)

img_trans = transform_train(img)
print(img.size)

np_img = np.asarray(img)
print(np_img.shape)

"""
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

"""

from torch.utils.data import Dataset, DataLoader
import glob

img_path_list = glob.glob("/tmp2/jojo/EyeJaundice/data/only_eyes/train/*/*.JPG")
print(len(img_path_list))

class MyDataset(Dataset):
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

train_data = MyDataset(img_path_list, transform_train)

data_loader = DataLoader(train_data, batch_size=32,shuffle=True)
print(len(data_loader))
