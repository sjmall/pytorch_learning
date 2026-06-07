import torch
torch.manual_seed(1)
"""
涉及的函数：
    torch.cat() 可以将多个张量根据指定的维度拼接起来，不改变维度数，除了拼接的那个维度，其他维度必须保持一致
    torch.stack() 可以在一个新的维度上连接一系列张量，这会怎么一个新的维度，并且所有输入张量的形状必须完全相同
"""
#场景一->cat()

t1 = torch.randint(1,10,(2,3))
t2 = torch.randint(1,10,(2,3))
print(f"t1: {t1}")
print(f"t2: {t2}")
t = torch.cat((t1,t2),0)
print(f"t: {t},t's shape: {t.shape}")
t = torch.cat((t1,t2),1)
print(f"t: {t},t's shape: {t.shape}")
print('-' * 50)

#场景二->stack()
#eg. t[i][0][j] = t1[i][j], t[i][1][j] = t2[i][j]
t1 = torch.randint(1,10,(2,3))
t2 = torch.randint(1,10,(2,3))
print(f"t1: {t1}")
print(f"t2: {t2}")
t = torch.stack((t1,t2),0)
print(f"t: {t},t's shape: {t.shape}")
t = torch.stack((t1,t2),1)
print(f"t: {t},t's shape: {t.shape}")
t = torch.stack((t1,t2),2)
print(f"t: {t},t's shape: {t.shape}")
print('-' * 50)