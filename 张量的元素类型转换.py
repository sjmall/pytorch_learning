import torch
t = torch.tensor([1,2,3,4,5])
print(f"t = {t},dtype = {t.dtype}")
print('-' * 30)
t = t.type(torch.float32)
print(f"t = {t},dtype = {t.dtype}")
print('-' * 30)

