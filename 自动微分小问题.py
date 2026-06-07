#不能将自动微分的张量转换成numpy数组，会发生报错，可以通过detach()函数实现
import torch
import numpy as np
#使用detach
t = torch.tensor([1,2],requires_grad=True,dtype=torch.float)
print(f"t = {t},t's type = {type(t)}")
t1 = t.detach()#detach得到的新张量共享空间！且不能自动微分
print(f"t1 = {t1},t1's type = {type(t1)}")
t1.data[0] = 100
print(f"t1 = {t1}")
print(f"t = {t}")
print(f"t's requires_grad = {t.requires_grad}")
print(f"t1's requires_grad = {t1.requires_grad}")
print('-' * 50)
