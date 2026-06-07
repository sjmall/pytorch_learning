"""
Dropout（随机失活）正则化
=========================

【原理】
    训练时以概率 p 随机丢弃（置零）一部分神经元，每次训练相当于采样一个不同的子网络，
    迫使网络不依赖特定神经元，增强泛化能力，防止过拟合。
    本质上是多个子网络的集成学习（bagging 效果）。

【做法】
    1. 只在训练时 Dropout，测试时所有神经元都参与
    2. PyTorch 的 nn.Dropout 使用 Inverted Dropout（训练时除以 1-p，测试直接输出）
    3. 一般在全连接层后使用，卷积层后用较少
    4. 常用概率 p = 0.2 ~ 0.5

【对比其他正则化】
    L1/L2 正则化：通过约束权重大小防止过拟合
    Dropout：通过随机丢弃神经元防止过拟合
    两者可配合使用


L1 正则化（Lasso）
==================

【原理】
    在损失函数中加入权重绝对值和（|w|），让权重趋向于零（稀疏化）。
    因为 L1 的导数恒为 ±1，每次更新权重都会减去一个固定值，可以精确收敛到0（类似辗转相除法）

【做法】
    loss = 原始损失 + λ * Σ|w|
    λ 越大，正则化越强，模型越稀疏

    在 PyTorch 中不能直接用参数设置，需手动在 loss 中加入 L1 惩罚项：

        l1_lambda = 1e-5
        l1_loss = sum(p.abs().sum() for p in model.parameters())
        loss = original_loss + l1_lambda * l1_loss

【优缺点】
    + 能产生稀疏解（很多权重为 0），自带特征选择
    - 导数不连续（在 0 处不可导），优化时可能不稳定
    - 使用不如 L2 广泛（因为没有内置支持）


L2 正则化（Ridge / Weight Decay）
==================================

【原理】
    在损失函数中加入权重平方和（w²），让权重大小趋向于均匀分布（权值衰减）。
    因为 L2 的导数为 2w，权重越大惩罚越大，迫使大权重变小，防止过拟合。

【做法】
    loss = 原始损失 + λ * Σw²

    PyTorch 中直接通过优化器的 weight_decay 参数设置（内置支持）：

        optimizer = optim.SGD(model.parameters(), lr=0.1, weight_decay=1e-4)

【优缺点】
    + 导数连续，优化稳定
    + PyTorch 内置支持，一行代码搞定
    - 不会产生稀疏解，权重只会接近零但不会精确为零
    - 需要调 weight_decay 参数（一般 1e-4 ~ 1e-2）


三种正则化对比总结
==================

| 方法         | 核心思想             | 效果                 | PyTorch 实现                   |
|-------------|---------------------|---------------------|--------------------------------|
| L1 正则化   | 对 |w| 惩罚           | 稀疏解（权重=0），特征选择 | 手动加到 loss 中             |
| L2 正则化   | 对 w² 惩罚           | 权重衰减，均匀分布     | optimizer 的 weight_decay 参数   |
| Dropout     | 随机丢弃神经元        | 集成学习，防过拟合     | nn.Dropout(p=0.5)              |

"""

"""
# ========== dropout小演示（已注释）==========
# x = torch.normal(2.0, 3.0, (1, 10))
# linear1 = nn.Linear(10, 5)
# output = torch.sigmoid(linear1(x))
# print(output)
# dropout = nn.Dropout(p=0.5)
# d = dropout(output)
# print(d)
"""

import torch
import torch.nn as nn

print("=" * 50)
print("Dropout（随机失活）正则化演示")
print("=" * 50)

# 1. 基本使用
print("\n【1. 基本使用】")
x = torch.normal(10.0, 3.0, size=(2, 4))
print("原始输入 x：")
print(x)

dropout = nn.Dropout(p=0.5)

dropout.train()
y_train = dropout(x)
print("\n训练模式（约一半被置零，剩余值放大）：")
print(y_train)

dropout.eval()
y_eval = dropout(x)
print("\n评估模式（Dropout 关闭，原样输出）：")
print(y_eval)

# 2. 嵌入神经网络
print("\n" + "=" * 50)
print("【2. 在神经网络中使用 Dropout】")
print("=" * 50)

class NetWithDropout(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 5)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(5, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x

net = NetWithDropout()
x2 = torch.randn(3, 10)

net.train()
print("\n训练模式前向传播：")
print(net(x2))

net.eval()
print("\n评估模式前向传播（关闭 Dropout）：")
print(net(x2))

print("\n注意：训练时用 net.train()，测试时用 net.eval()")
