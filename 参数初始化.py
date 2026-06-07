"""
m,in是输入该层的神经元的数量，m,out是这层神经元的数量
1.Lecun初始化
    正态分布N(0,1/m,in) 均匀分布U(-sqrt(3/m,in),sqrt(3/m,in))
2.Xavier初始化
    正态分布N(0,2/(m,in+m,out)) 均匀分布U(-sqrt(6/(m,in+m,out)),sqrt(6/(m,in+m,out)))
    对Sigmoid函数，Tanh函数效果好，对ReLU函数不好，适合深层网络
3.Kaiming初始化
    正态分布N(0,2/m,in) 均匀分布U(-sqrt(6/m,in),sqrt(6/m,in))
    特别的，Kaiming初始化针对ReLU及其变体设计，对其他激活函数效果不佳，适合深层网络
注意上诉初始化公式不是绝对的，根据不同的激活函数会有不同的系数
"""
import torch
import torch.nn as nn
linear = nn.Linear(5, 3)
torch.manual_seed(42)
#场景一->0 to 1均匀分布初始化
nn.init.uniform_(linear.weight)
nn.init.uniform_(linear.bias)
print(linear.weight)
print(linear.bias)
print("-" * 50)

#场景二->相同初始化
nn.init.constant_(linear.weight,3)
print(linear.weight)
nn.init.zeros_(linear.weight)
print(linear.weight)
nn.init.ones_(linear.weight)
print(linear.weight)
print("-" * 50)

#场景三->均值0，标准差1的正态分布初始化
nn.init.normal_(linear.weight,mean=0,std=1)#默认也是均值0，标准差1
print(linear.weight)
print('-' * 50)

#Kaiming初始化 默认激活函数是ReLU
nn.init.kaiming_normal_(linear.weight)
print(linear.weight)
nn.init.kaiming_uniform_(linear.weight)
print(linear.weight)
print("-" * 50)

#Xavier初始化
nn.init.xavier_normal_(linear.weight)
print(linear.weight)
nn.init.xavier_uniform_(linear.weight)
print(linear.weight)
print("-" * 50)