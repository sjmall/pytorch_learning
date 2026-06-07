"""
分类问题：
    1.多分类交叉熵损失：CrossEntropyLoss 内部自带softmax
    2.二分类交叉熵损失：BCELoss 内部不带sigmoid
回归问题：
    1.MAE：Mean Absolute Error 平均绝对损失
    2.MSE；Mean Squared Error
    3.Smooth L1：结合以上两点的升级

正则项->防止权重过大（意味着模型对输入敏感，复杂度高）
L1正则化采取在小于阈值会直接为0的策略，容易使权重稀疏
L2正则化在小于阈值后只是乘以小于1的系数而不会为0
"""
import torch
import torch.nn as nn

# #场景一->多分类交叉熵损失计算
# y_true = torch.tensor([1,2,0])#表明每个样本属于哪个类别，由于不参与运算所以不需要是float型
# y_pred = torch.tensor([[3,20,2],[1,2,31],[12,2,3]],dtype=torch.float,requires_grad=True)
# criterion = nn.CrossEntropyLoss()#注意该函数会自动对数据进行softmax处理
#
# loss = criterion(y_pred,y_true)
# print(loss)

# #场景二->二分类交叉熵损失
# y_true = torch.tensor([1,1,0],dtype=torch.float)#这里的标签参与运算，要求是float型
# y_pred = torch.tensor([1,-2,1],dtype=torch.float,requires_grad=True)#每个元素代表一个样本预测正类的输出值
# criterion = nn.BCELoss()
# y_pred = torch.sigmoid(y_pred)
# loss = criterion(y_pred,y_true)
# print(loss)

# #场景三->MAE，MSE，SmoothL1
# y_pred = torch.tensor([1,2,3],dtype=torch.float)
# y_ture = torch.tensor([4,5,6],dtype=torch.float)
# criterion1 = nn.L1Loss()
# loss = criterion1(y_pred,y_ture).detach().numpy()
# print(loss)
# criterion2 = nn.MSELoss()
# loss = criterion2(y_pred,y_ture).detach().numpy()
# print(loss)
# criterion3 = nn.SmoothL1Loss()
# loss = criterion3(y_pred,y_ture).detach().numpy()
# print(loss)