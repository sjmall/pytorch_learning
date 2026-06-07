"""
图像知识点：
    1.图像的像素点�?->255，越接近0越黑，越接近255越白
      正常彩色图片包含RGB三个通道，每个通道都有0->255的�?
    2.计算机中可以把图像分�?个类�?
        第一类：二值图像，由二维矩阵构成，只有0�?两个值，0代表纯黑�?代表纯白�?
        第二类：灰度图像，也是二维矩阵，但是取值从0�?55，是从黑色到白色的过渡，只有一个通道
        第三类：索引图象，除了存放图像的二维矩阵外，还包括一个称之为颜色索引矩阵MAP的二维数组，
               存放图像的二维矩阵中的元素值是MAP矩阵的第几行，MAP矩阵的三个列分别对应红绿蓝三个单元色的单�?
               此类图像只有一个通道
        第四类：正彩色RGB图像，有R，G，B三个通道
"""
import numpy as np
import matplotlib.pyplot as plt
import torch

#场景一->绘制全黑，全白图
def dm01():
    img1 = np.zeros((256, 256, 3))#plt根据shape来绘图，第一个位置是H：高度；第二个位置是W：宽度；第三个位置是C：通道�?
    plt.imshow(img1)
    plt.show()
    img2 = torch.full((256, 256, 3), 255, dtype=torch.uint8)
    plt.imshow(img2)
    plt.show()

#场景二->加载图片
def dm02():
    img = plt.imread("./data/photos/mywife.jpg")
    plt.imshow(img)
    plt.show()
    plt.imsave("./data/photos/mywife_copy.jpg", img)

#测试
dm01()
dm02()
