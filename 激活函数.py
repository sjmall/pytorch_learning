"""
1.Sigmoid函数
    1.输入值在[-6,6]之间才会有明显的差异，在[-3,3]之间才会有比较好的效果
    2.特别是当输入在[-6,6]之间的时候其导数值接近于0，容易无法更新
    3.由于其导数值最大值是0.25，一般在五层左右就会出现梯度消失，其一般用于二分类的输出层，输出正类的概率

2.tanh函数
    1.y' = 1 - y^2, y = (e^x-e^(-x))/(e^x+e^(-x))
    2.中心点在(0,0)处，在[-3,3]之间有比较明显的差异，函数值在(-1,1)，导数值在(0,1)
    3.其梯度相对于sigmoid较大，迭代速度快
    4.其主要用于浅层的神经网络

3.ReLU函数
    1.计算速度快，能用于深层网络，且收敛速度快
    2.一般用于隐藏层

4.softmax函数
    1.一般用于多二分类问题
    2.具有平移不变性


隐藏层激活函数：ReLU->leakyReLU->PReLU->Tanh->Sigmoid
"""
import torch
import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体（避免乱码和警告）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

# def sigmoid(x):
#     """Sigmoid 函数定义"""
#     return 1 / (1 + np.exp(-x))
#
# # 生成 x 轴数据（-10 到 10 之间的 200 个点）
# x = np.linspace(-10, 10, 200)
# y = sigmoid(x)
#
# # 创建图形
# plt.figure(figsize=(8, 5))
# plt.plot(x, y, linewidth=2, color='blue', label='Sigmoid')
#
# # 添加参考线
# #axh表示纵坐标，axv表示横坐标，‘--’表示虚线，‘：’表示点线
# plt.axhline(0, color='black', linewidth=0.5, linestyle='--')   # y=0
# plt.axhline(0.5, color='gray', linewidth=1, linestyle=':')     # 中间值 y=0.5
# plt.axhline(1, color='black', linewidth=0.5, linestyle='--')   # y=1
# plt.axvline(0, color='black', linewidth=0.5, linestyle='--')   # x=0
#
# # 标注关键点
# #‘ro’表示圆形红点，fontsize表示字体大小
# plt.plot(0, 0.5, 'ro')  # 中心点 (0, 0.5)
# plt.text(0.3, 0.45, '(0, 0.5)', fontsize=10)
#
# # 设置坐标轴范围和标签
# #xlim,ylim表示坐标轴数值范围
# plt.xlim(-10, 10)
# plt.ylim(-0.05, 1.05)
# plt.xlabel('x', fontsize=12)
# plt.ylabel('σ(x)', fontsize=12)
# plt.title('Sigmoid 函数图像', fontsize=14)
# #把图例放在左上角，loc是图例的位置
# plt.legend(loc='upper left')
# #alpha表示网格线的透明程度，0为完全透明，1为完全不透明
# plt.grid(True, alpha=0.3)
#
# # 显示图像
# plt.tight_layout()#自动调整子图周围的空白边距，确保所有标签、标题等不会被裁剪或重叠。
# plt.show()


# fig, axes = plt.subplots(1,2)
"""
fig：整个图形窗口（Figure 对象），可以用来设置标题、大小、保存等全局属性。

axes：子图坐标轴的容器。因为这里是 1 行 2 列，axes 是一个包含 2 个 Axes 对象的 NumPy 数组（形状为 (2,)）。
      你可以通过 axes[0] 访问第一个子图，axes[1] 访问第二个子图。
"""
# fig.suptitle("Tanh")
# x = torch.linspace(-20,20,100,requires_grad=True)
# y = torch.tanh(x)
#
# axes[0].plot(x.detach(),y.detach())#注意matplotlib的参数不能是能自动微分的张量，由自动微分的张量计算出的张量也能微分，但是其grad不能
# axes[0].set_title('Tanh激活函数图像')
# axes[0].grid()
#
# y.sum().backward()
#
# axes[1].plot(x.detach(),x.grad)
# axes[1].set_title('Tanh激活函数导数图像')
# axes[1].grid()
# plt.show()

t = torch.tensor([[1.,2 ,3 ,5],
                 [1 ,4 ,2 ,3]])
pro1 = torch.softmax(t,dim=1)
print(pro1)
pro2 = torch.softmax(t,dim=0)
print(pro2)