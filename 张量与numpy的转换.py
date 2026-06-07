"""
一->张量转换为ndarray
可以使用Tensor.numpy函数将张量转换为ndarray数组，但是是共享内存，可以使用copy函数进行深拷贝
二->ndarray转化为张量
有函数torch.from_numpy，或者torch.tensor(Tensor)，前者是浅拷贝，后者是深拷贝
"""
import torch
import numpy as np
#场景一->张量转numpy
t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(id(t))
t1 = t.numpy()#共享内存只是二者对象里面的指针指向相同，但是对象本身的地址还是不一样
print(f"t1 = {t1}, type(t1) = {type(t1)},id(t1) = {id(t1)}")
t2 = t.numpy().copy()
print(f"t2 = {t2}, type(t2) = {type(t2)},id(t2) = {id(t2)}")
t[0][0] = 6
print(f"修改了t[0][0] = {t[0][0]}")
print(f"修改之后t1[0][0] = {t1[0][0]}")
print('-' * 30)
#场景二->numpy转换成张量
t = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
t1 = torch.tensor(t)
print(f"t1 = {t1}, type(t1) = {type(t1)}")
t2 = torch.from_numpy(t)
print(f"t2 = {t2}, type(t2) = {type(t2)}")
print('-' * 30)
#场景三->从张量中提取值,张量只能由一个值
t = torch.tensor([1,])
data = t.item()
print(f"t = {t}, type(t) = {type(t)}, data = {data}")#深拷贝
print('-' * 30)