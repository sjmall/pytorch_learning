"""
三部分组成：1.卷积层：提取图像特征
          2.池化层：用来大幅降低参数量级（降维）
          3.全连接层：全连接层用来输出想要的结果
填充Paddinng：用来防止卷积后图片缩水，同时加重边缘信息提取（使得其能被多算几次）
    Padding类型：
        1.Valid Padding：不进行任何填充
        2.Same Padding；添加足够多的填充使得卷积后图片大小不变
        3.Full Padding：尽可能多的填充，使得卷积核中每个位置都能经过图片中的每个位置
步长Stride：能在多层卷积后增大感受野，减少计算量
多通道卷积核：对应通道点乘，再相加得到单通道的结果，注意多通道卷积核不同通道的参数可以不同
同样对于多通道图一般配有多个多通道卷积核，卷积后得到通道数等于卷积核数的图片

池化Pooling：池化一般不改变通道数，在处理多通道输入数据的时候，池化只是对不同通道分别做相同的池化

API:
    conv = nn.Conv2d(in_channels, out_channels, kernel_size//卷积核的大小, stride, padding)、
    nn.MaxPool2d(kernel_size, stride, padding)
    nn.AvgPool2d(kernel_size, stride, padding)
"""
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

#场景一->运用卷积核对图片卷据
def dm01():
    img = plt.imread("./data/photos/mywife_copy.jpg")
    print(f"img = {img},img.shape = {img.shape}")#内部是numpy数组
    print('-' * 55)

    img1 = torch.tensor(img).float().permute(2, 0, 1)
    print(f"img0 = {img1},img0.shape = {img1.shape}")
    print('-' * 55)

    img2 = img1.unsqueeze(dim=0)
    print(f"img2 = {img2},img2.shape = {img2.shape}")
    print('-' * 55)

    conv = nn.Conv2d(3,4,3)
    conv_img = conv(img2)#其接受四维张量，(图片数,通道数,行数,列数)，由于卷积核可自动微分，这里的结果也自然可以自动微分
    print(f"conv_img = {conv_img},conv_img.shape = {conv_img.shape}")
    print('-' * 55)

    img4 = conv_img[0].permute(1,2,0)
    feture = img4[:, :, 3].detach().numpy()
    plt.imshow(feture)
    plt.show()

#场景二->演示最大值池化和平均池化
def dm02():

    img = plt.imread("./data/photos/mywife_copy.jpg")
    img1 = torch.tensor(img).float().permute(2, 0, 1)
    print(f"img1.shape = {img1.shape}")
    print('-' * 55)
    img2 = img1.unsqueeze(dim=0)
    pool1 = nn.MaxPool2d(33, stride=1, padding=16)
    pool2 = nn.AvgPool2d(13, stride=1, padding=6)
    result1 = pool1(img1)
    result2 = pool2(img1)
    print(f"result1.shape = {result1.shape}")
    print(f"result2.shape = {result2.shape}")
    result1 = result1.int().permute(1, 2, 0)
    result2 = result2.int().permute(1, 2, 0)
    plt.imshow(result1)
    plt.show()
    plt.imshow(result2)
    plt.show()

#测试
dm02()