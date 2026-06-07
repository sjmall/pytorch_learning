"""
核心参数：
    step_size:更新多少次调整一次学习率
    gamma:学习率衰减系数

"""
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
# 三种学习率衰减策略优缺点总结：
#   StepLR（等间隔衰减）→ 优点：实现简单，参数少，每 step_size 轮统一衰减，行为可预测；缺点：衰减时机固定，不够灵活，无法针对不同阶段调整
#   MultiStepLR（指定间隔衰减）→ 优点：可在关键节点（milestones）手动调整，灵活可控，适合分阶段训练；缺点：需要人工经验设定 milestones，调参成本高
#   ExponentialLR（指数衰减）→ 优点：每轮平滑衰减，初期快速下降后期微调，曲线连续自然；缺点：衰减过快可能导致后期学习率过小，收敛停滞

lr, epochs = 0.1, 200
y_true = torch.tensor([0],dtype=torch.float)
w = torch.tensor([1],dtype=torch.float,requires_grad=True)
x = torch.tensor([1],dtype=torch.float)
optimizer = optim.SGD([w], lr=lr, momentum=0.9)
#创建学习率衰减对象
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)#StepLR等间隔衰减
# scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,30,50,100,200], gamma=0.5)#MultiStepLR指定间隔衰减
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)#ExponentialLR指数衰减

epoch_list = []
lr_list = []
criterion = nn.MSELoss()
for epoch in range(epochs):
    epoch_list.append(epoch+1)
    lr_list.append(scheduler.get_last_lr())
    optimizer.zero_grad()
    output = w * x
    loss = criterion(output, y_true)
    loss.backward()
    optimizer.step()
    scheduler.step()#学习率衰减对象本身也需要更新

plt.plot(epoch_list,lr_list,label='lr_value',color='blue',linewidth=1,linestyle='-',marker='*')
plt.xlabel('epoch')
plt.ylabel('lr')
plt.legend()
plt.grid(True, alpha = 0.5)
plt.show()