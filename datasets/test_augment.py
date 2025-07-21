import numpy as np
from PIL import Image, ImageFilter, ImageEnhance

import random
import cv2

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


def add_gaussian_blur(image):
    # 将PIL图片转换为OpenCV格式
    image_np = np.array(image)

    # 随机生成高斯核的大小，通常核大小是正奇数
    kernel_size = (random.randint(3, 11), random.randint(3, 11))

    # 随机生成高斯核的标准差
    sigma_x = random.uniform(0, 5)
    sigma_y = random.uniform(0, 5)

    # 应用高斯模糊
    blurred_image_np = cv2.GaussianBlur(image_np, kernel_size, sigmaX=sigma_x, sigmaY=sigma_y)

    # 将OpenCV格式转换回PIL图片
    blurred_image = Image.fromarray(blurred_image_np)

    return blurred_image


def add_gaussian_noise(image):
    # 将图片转换为numpy数组
    image_np = np.array(image)

    # 定义噪声的标准差
    mean = 0
    sigma = random.uniform(0.5, 5.0)  # 随机选择一个标准差值

    # 生成高斯噪声
    gaussian_noise = np.random.normal(mean, sigma, image_np.shape)

    # 将噪声添加到图片数组上
    noisy_image_np = image_np + gaussian_noise

    # 将图片数组的值限制在0到255之间
    noisy_image_np = np.clip(noisy_image_np, 0, 255)

    # 将处理后的numpy数组转换回图片对象
    noisy_image = Image.fromarray(np.uint8(noisy_image_np))

    return noisy_image


# 使用示例
# 假设你有一个PIL Image对象名为original_image
# noisy_image = add_gaussian_noise(original_image)
# noisy_image.show()  # 显示添加噪声后的图片

def addGaussianNoise(image, mean=0, var=0.5):
    """
    对输入图像添加高斯噪声。

    :param image: PIL.Image 对象
    :param mean: 噪声的均值（默认为0）
    :param var: 噪声的方差（默认为0.1）
    :return: 添加噪声后的 PIL.Image 对象
    """
    # 将图像转换为 numpy 数组
    img_array = np.array(image)

    # 生成高斯噪声
    noise = np.random.normal(mean, var ** 0.5, img_array.shape)

    # 将噪声添加到图像中
    noisy_img_array = img_array + noise

    # 将像素值限制在0到255之间
    noisy_img_array = np.clip(noisy_img_array, 0, 255).astype(np.uint8)

    # 将 numpy 数组转换回 PIL 图像
    noisy_image = Image.fromarray(noisy_img_array)

    return noisy_image


def rgb_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


# 调用 rgb_loader 函数加载图像
img = rgb_loader('test.jpg')

# 水平翻转图像
# flipped_img = img.transpose(Image.FLIP_LEFT_RIGHT)

# 展示翻转后的图像
# flipped_img.show()
# enhancer = ImageEnhance.Brightness(img)
# enhanced_image = enhancer.enhance(1.5)  # 增强因子为1.5，可以根据需要调整
enhanced_image = colorEnhance(randomPeper(img))
# 显示增强后的图像
enhanced_image.show()
