#张量->存储同一元素类型的容器，且元素必须是数值
import torch
import numpy as np
"""
1.tensor是根据指定诗句来创建张量
2.Tensor是根据指定数据或者指定形状来创建张量
3.张量的0维表示一个数，一维表示一个向量，二维...
"""
# #场景一->创建标量张量
# t1 = torch.tensor(10)
# print(f"t1 = {t1}, type is {type(t1)}")
# print("-"*30)
# #场景二->根据二维列表创建张量
# data = [[1,2,3],[1,2,3]]
# t2 = torch.Tensor(data)
# print(f"t2 = {t2}, type is {type(t2)}")
# print("-"*30)
# #场景三->利用numpy来创建张量
# data = np.random.randint(0,10,size = (2,3))
# t3 = torch.Tensor(data)
# print(f"t3 = {t3}, type is {type(t3)}")
# print("-"*30)
# #场景四->根据指定形状来创建张量
# t4 = torch.Tensor(2,3)
# print(f"t4 = {t4}, type is {type(t4)}")
# print("-"*30)

# #还有IntTensor,FloatTensor,DoubleTensor,当传入数据时，如果类型不匹配会尝试类型转换
# t1 = torch.IntTensor(2,3)
# t2 = torch.FloatTensor(2,3)
# t3 = torch.DoubleTensor(2,3)
# print(t1)
# print(t2)#float是默认类型，print时不会打印元素类型
# print(t3)
# print('-'*30)

# #创建全1，全0，全为指定值的张量
# #全1
# t1 = torch.ones(2,3)#根据形状,元素类型为float32
# print(f"t1 = {t1},type(t1) = {type(t1)},dtype = {t1.dtype}")
# print("-"*30)
# t = torch.tensor([[1,2,3],[4,5,6]])#根据传入数据的形状和元素类型
# t3 = torch.ones_like(t)
# print(f"t3 = {t3},type(t3) = {type(t3)},dtype = {t3.dtype}")
# print("-"*30)
# #全0
# t1 = torch.zeros(2,3)#根据形状,元素类型为float32
# print(f"t1 = {t1},type(t1) = {type(t1)},dtype = {t1.dtype}")
# print("-"*30)
# t = torch.tensor([[1,2,3],[4,5,6]])#根据传入数据的形状和元素类型
# t3 = torch.zeros_like(t)
# print(f"t3 = {t3},type(t3) = {type(t3)},dtype = {t3.dtype}")
# print("-"*30)
# #全为指定值
# t1 = torch.full((2,3),4.0)
# print(f"t1 = {t1},type(t1) = {type(t1)},dtype = {t1.dtype}")
# print("-"*30)
# t = torch.tensor([[1,2,3],[4,5,6]])
# t3 = torch.full_like(t,3.9)#注意虽然指定值是3.9，但是由于元素类型由传入数据决定，所以3.9会转化为int，即为3
# print(f"t3 = {t3},type(t3) = {type(t3)},dtype = {t3.dtype}")
# print("-"*30)

# #创建线性张量
# #场景一->创建指定范围的线型张量
# t = torch.arange(0,10,2)#不包括end，默认int
# print(f"t = {t},dtype = {t.dtype}")
# print('-'*30)
# #场景二->创造指定范围的等差数列一维张量
# t = torch.linspace(1,10,3)
# print(f"t = {t},dtype = {t.dtype}")#默认float32,step代表分成几分，可以包括end
# print('-'*30)

# #创建随机张量
# #场景一->用系统时间戳作为随机种子
# torch.initial_seed()
# t = torch.rand((2,3))#rand生成0到1均匀分布的随机数
# print(f"t = {t}, type(t) = {type(t)}")
# print('-'*30)
# #场景二->使用固定随机种子
# torch.manual_seed(71910025)
# t = torch.rand((2,3))
# print(f"t = {t}, type(t) = {type(t)}")
# print('-'*30)
# #场景三->生成正态分布的随机张量
# torch.manual_seed(3)
# t = torch.randn((2,3))
# print(f"t = {t}, type(t) = {type(t)}")
# print('-'*30)
# #场景四->生成指定范围随机整数的张量
# t = torch.randint(1,10,(2,3))#不会取到max
# print(f"t = {t}, type(t) = {type(t)}")
# print('-'*30)































