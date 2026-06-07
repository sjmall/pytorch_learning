"""
加减乘除取负号
函数：add sub mul div neg
     add_ sub_ mul_ div_ neg_
     其中带_的会改变原数据
但实际上只需要用+ - * /替换就可以
"""
import torch
# t1 = torch.tensor([1,2,3,4,5,6]).type(torch.int32)
# t2 = torch.tensor([1,2,3,4,5,6])
# t = t1 + t2
# print(f"t = t1 + t2 = {t}")
# t = t1.neg_()
# print(f"t = -t1 = {t}，函数是neg_")
# print(f"t1 = {t1}")
# print('-' * 50)
# t = t1 / 2 #除法会默认改变二者的元素类型为float型
# print(f"t = {t}，t1.dtype = {t.dtype},t.dtype = {t.dtype}")

# #矩阵点乘，要求shape一样
# t1 = torch.tensor([[1,2,3],[4,5,6],[7,8,9]])
# t2 = t1
# t = t1 * t2
# print(f"t = t1 * t2 = {t}")
# print('-' * 50)

# #矩阵乘法 @ or Tensor.matmul()
# t1 = torch.tensor([[1,2,3],[4,5,6]])
# t2 = torch.tensor([[7,8],[10,11],[12,13]])
# t = t1 @ t2 #or t = t1.matmul(t2)
# print(f"t = t1 @ t2 = {t}")
# print('-' * 50)

# #一维张量的内积 Tensor.dot()
# t1 = torch.tensor([1,2,3,4,5,6])
# t2 = torch.tensor([1,2,3,4,5,6])
# t = t1.dot(t2)
# print(f"t = t1·t2 = {t}")