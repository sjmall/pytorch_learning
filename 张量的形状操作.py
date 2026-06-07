import torch
torch.manual_seed(1)
"""
涉及到的函数：
    reshape()
    unsqueeze()
    squeeze()
    transpose()
    permute()
    view()
    contiguous()
    is_contiguous()
"""
#场景一->reshape function
#注意reshape返回的是视图，但如果张量内元素的内存不连续，reshape就会返回视图
t = torch.randint(1,10,(2,3))
print(f"t: {t},t's shape: {t.shape}")
t1 = t.reshape(3,2)
print(f"t1: {t1},t's shape: {t1.shape}")
t2 = t.reshape(1,6)
print(f"t2: {t2},t's shape: {t2.shape}")
print('-' * 50)

#场景二->unsqueeze function 在指定的轴上增加一个为1的维度，指定位置不超过张量维度+1 返回视图
t = torch.randint(1,10,(2,3))
print(f"t: {t},t's shape: {t.shape}")
t1 = t.unsqueeze(0)
print(f"t1: {t1},t's shape: {t1.shape}")
t2 = t.unsqueeze(1)
print(f"t2: {t2},t's shape: {t2.shape}")
t3 = t.unsqueeze(2)
print(f"t3: {t3},t's shape: {t3.shape}")
print('-' * 50)

#场景三->squeeze function 删除所有为1的维度 返回视图
t = torch.randint(1,10,(2,1,3,1,1))
print(f"t: {t},t's shape: {t.shape}")
t = t.squeeze()
print(f"t: {t},t's shape: {t.shape}")
print('-' * 50)

#场景四->transpose function 一次只能交换两个维度 返回视图
#t[i][j][k] = new_t[j][i][k] if exchange dim-0,dim-1 一般会让张量的内存保存不连续
t = torch.randint(1,10,(2,3))
print(f"t: {t},t's shape: {t.shape}")
t1 = t.transpose(0, 1)
print(f"t1: {t1},t's shape: {t1.shape}")
print('-' * 50)

#场景五->permute function transpose的升级版
t = torch.randint(1,10,(2,3,4))
print(f"t: {t},t's shape: {t.shape}")
t1 = t.permute(2, 0, 1)
print(f"t1: {t1},t's shape: {t1.shape}")
print('-' * 50)

#场景六->view function 只能修改内存连续的张量 效果等同于reshape
t = torch.randint(1,10,(2,3))
print(f"t: {t},t's shape: {t.shape}")
print(t.is_contiguous())#判断张量的内存是否连续
t.transpose_(0, 1)
print(t.is_contiguous())
t = t.contiguous()#将t转化为连续的张量，如果张量已经连续，则返回自身，不然，返回副本
print(t.is_contiguous())
print('-' * 50)

