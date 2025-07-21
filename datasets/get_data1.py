import os
import random

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image, ImageFilter
from PIL import ImageEnhance

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


def randomPeper(img):
    img = np.array(img)
    noiseNum = int(0.085 * img.shape[0] * img.shape[1])
    for i in range(noiseNum):

        randX = random.randint(0, img.shape[0] - 1)

        randY = random.randint(0, img.shape[1] - 1)

        if random.randint(0, 1) == 0:

            img[randX, randY] = 0

        else:

            img[randX, randY] = 255
    return Image.fromarray(img)


def blur(img, p=0.5):
    if random.random() < p:
        sigma = np.random.uniform(0.1, 2.0)
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
    return img


def colorEnhance(image):
    bright_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Brightness(image).enhance(bright_intensity)
    contrast_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Contrast(image).enhance(contrast_intensity)
    color_intensity = random.randint(0, 20) / 10.0
    image = ImageEnhance.Color(image).enhance(color_intensity)
    sharp_intensity = random.randint(0, 30) / 10.0
    image = ImageEnhance.Sharpness(image).enhance(sharp_intensity)
    return image


def get_prompt(x, y, points_num, imgsize):
    point_list = []
    label_list = []
    t = 0
    while len(point_list) < points_num and t < 100000:
        t = t + 1
        random_x = np.random.randint(0, imgsize)
        random_y = np.random.randint(0, imgsize)
        x_value = x.getpixel((random_x, random_y))
        y_value = y.getpixel((random_x, random_y))
        # 判断前景或者背景
        if x_value == 255 and y_value == 255:
            point_list.append([random_x, random_y])
            label_list.append(1)
        elif x_value == 0 and y_value == 255:
            point_list.append([random_x, random_y])
            label_list.append(0)

    # if len(point_list) < 1:
    #     point_list.append([0, 0])
    #     label_list.append(-1)

    while len(point_list) < 20:
        point_list.append([0, 0])
        label_list.append(-1)

    # 将随机选择的点坐标转换为tensor张量作为输入点
    input_point = torch.tensor(point_list)
    # 定义对应输入点的标签
    input_label = torch.tensor(label_list)
    return input_point, input_label


class SalObjDataset(data.Dataset):
    def __init__(
            self, image_root, any_modal_root, gt_root, trainsize
    ):
        self.trainsize = trainsize
        self.images = [os.path.join(image_root, f) for f in os.listdir(image_root) if
                       f.endswith('.jpg') or f.endswith('.png')]
        self.any_modal = [os.path.join(any_modal_root, f) for f in os.listdir(any_modal_root) if
                          f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [os.path.join(gt_root, f) for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.images = sorted(self.images)
        self.any_modal = sorted(self.any_modal)
        self.gts = sorted(self.gts)
        self.size = len(self.images)

        self.resize_transform = transforms.Resize((self.trainsize, self.trainsize))
        self.to_tensor_transform = transforms.ToTensor()
        self.normalize_transform = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        self.up = transforms.Resize((self.trainsize, self.trainsize),
                                    interpolation=transforms.InterpolationMode.NEAREST)
        self.points_num = 20

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        aug_image = colorEnhance(image)
        any_modal = self.rgb_loader(self.any_modal[index])
        aug_modal = colorEnhance(any_modal)
        gt = self.binary_loader(self.gts[index])
        aug_image = self.normalize_transform(self.to_tensor_transform(self.resize_transform(aug_image)))
        image = self.normalize_transform(self.to_tensor_transform(self.resize_transform(image)))

        # 针对不是rgb模态的情况
        any_modal = self.to_tensor_transform(self.resize_transform(any_modal))
        aug_modal = self.to_tensor_transform(self.resize_transform(aug_modal))

        gt = self.to_tensor_transform(self.resize_transform(gt))
        return image, aug_image, any_modal, aug_modal, gt

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size


class SalObjDataset_rgb(data.Dataset):
    def __init__(
            self, image_root, any_modal_root, gt_root, trainsize
    ):
        self.trainsize = trainsize
        self.images = [os.path.join(image_root, f) for f in os.listdir(image_root) if
                       f.endswith('.jpg') or f.endswith('.png')]
        self.any_modal = [os.path.join(any_modal_root, f) for f in os.listdir(any_modal_root) if
                          f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [os.path.join(gt_root, f) for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.images = sorted(self.images)
        self.any_modal = sorted(self.any_modal)
        self.gts = sorted(self.gts)
        self.size = len(self.images)

        self.resize_transform = transforms.Resize((self.trainsize, self.trainsize))
        self.to_tensor_transform = transforms.ToTensor()
        self.normalize_transform = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        self.up = transforms.Resize((self.trainsize, self.trainsize),
                                    interpolation=transforms.InterpolationMode.NEAREST)
        self.points_num = 20

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        # 翻转增强
        aug_image = image.transpose(Image.FLIP_LEFT_RIGHT)

        # 亮度增强
        aug_image = ImageEnhance.Brightness(aug_image)
        aug_image = aug_image.enhance(1.5)
        aug_image = ImageEnhance.Contrast(aug_image)
        aug_image = aug_image.enhance(1.5)

        any_modal = self.rgb_loader(self.any_modal[index])

        gt = self.binary_loader(self.gts[index])
        aug_image = self.normalize_transform(self.to_tensor_transform(self.resize_transform(aug_image)))
        image = self.normalize_transform(self.to_tensor_transform(self.resize_transform(image)))
        any_modal = self.normalize_transform(self.to_tensor_transform(self.resize_transform(any_modal)))

        # 针对不是rgb模态的情况
        # any_modal = self.to_tensor_transform(self.resize_transform(any_modal))

        gt = self.to_tensor_transform(self.resize_transform(gt))
        return image, aug_image, gt
        # return image, depth, gt, mask, gray

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size


def get_loader(
        image_root, any_modal_root, gt_root, batchsize, trainsize,
        shuffle=True, num_workers=0, pin_memory=True
):
    dataset = SalObjDataset(
        image_root, any_modal_root, gt_root, trainsize,
    )
    data_loader = data.DataLoader(
        dataset=dataset, batch_size=batchsize, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory
    )
    return data_loader


def get_loader_rgb(
        image_root, any_modal_root, gt_root, batchsize, trainsize,
        shuffle=True, num_workers=0, pin_memory=True
):
    dataset = SalObjDataset_rgb(
        image_root, any_modal_root, gt_root, trainsize,
    )
    data_loader = data.DataLoader(
        dataset=dataset, batch_size=batchsize, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory
    )
    return data_loader


class test_dataset:
    def __init__(self, image_root, any_modal_root, gt_root, testsize):
        self.testsize = testsize
        self.path = any_modal_root
        self.images = [image_root + f for f in os.listdir(image_root) if
                       f.endswith('.jpg') or f.endswith('.png') or f.endswith('.bmp')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.any_modal_root = [any_modal_root + f for f in os.listdir(any_modal_root) if f.endswith('.bmp')
                               or f.endswith('.jpg') or f.endswith('.png') or f.endswith('.tiff')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.any_modal_root = sorted(self.any_modal_root)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.ToTensor()
        self.depths_transform = transforms.Compose(
            [transforms.Resize((self.testsize, self.testsize)), transforms.ToTensor()])
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])

        enhancer = ImageEnhance.Brightness(image)
        image_enhance = enhancer.enhance(1.5)

        image = self.transform(image).unsqueeze(0)
        image_enhance = self.transform(image_enhance).unsqueeze(0)
        gt = self.binary_loader(self.gts[self.index])
        any_modal = self.rgb_loader(self.any_modal_root[self.index])

        # any_modal = self.transform(any_modal).unsqueeze(0)

        # 针对不是rgb模态的情况
        any_modal = self.depths_transform(any_modal).unsqueeze(0)

        name = self.images[self.index].split('/')[-1]
        if name.endswith('.bmp'):
            name = name.split('.bmp')[0] + '.png'
        if name.endswith('.jpg'):
            # name = name.split('.jpg')[0] + '.png'
            name = name.split('.jpg')[0] + '.jpg'
        self.index += 1
        self.index = self.index % self.size
        PATH = self.path + name

        return image, PATH, gt, any_modal, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

