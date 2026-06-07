"""
涉及的函数
1.sum(),max(),min(),mean() 含有dim属性，dim与axis相同，其取值表示shape的第几个位置
2.pow(),sqrt(),exp(),log(),log2(),log10() 不含dim属性
"""
import torch
#场景一->带dim参数的函数
t = torch.tensor([[1,2,3],[4,5,6]],dtype = torch.float)
print(f"t = {t}")
print(f"t's sum of dim = 0 is {t.sum(dim = 0)}")#注意返回的是张量
print(f"t's sum of all is {t.sum()}")
print('-' * 30)
print(f"t's mean = {t.mean(dim = 1)}")#注意mean函数要求dtype是float型
print('-' * 30)
#场景二->不带dim参数的函数
print(f"t^3.1 = {t.pow(3.1)}")
print('-' * 30)
print(f"t^0.5 = {t.sqrt()}")
print('-' * 30)
print(f"lnt = {t.log()}")
print('-' * 30)
print(f"e^t = {t.exp()}")
print('-' * 30)
