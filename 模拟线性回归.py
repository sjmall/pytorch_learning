import torch
from torch.utils.data import TensorDataset      #构造数据集对象
from torch.utils.data import DataLoader         #数据加载器
from torch import nn                            #nn模块中有平方损失函数和假设函数
from torch import optim                         #optim模块中有优化器函数
from sklearn.datasets import make_regression    #创建线性回归模型数据集
import matplotlib.pyplot as plt                 #可视化

plt.rcParams['font.sans-serif'] = ['SimHei']    #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False      #用来正常显示负号

"""
转换流程：
    numpy对象->张量Tensor->数据集对象TensorDataset->数据加载器DataLoader
"""

#1.创建线性回归样本数据
def create_dataset():
    #注意x，y是numpy.ndarray对象
    x, y, coef = make_regression(
        n_samples = 100,
        n_features = 1,
        noise = 10,
        coef = True,
        random_state = 42,
        bias = 14
    )
    x = torch.tensor(x,dtype=torch.float)
    y = torch.tensor(y,dtype=torch.float)
    print(type(coef))
    coef = coef.item()#coef是numpy数组，当该类型与张量相乘会触发警告
    return x,y,coef

#2.训练模型
def train(x, y, coef):
    #1.创建数据集对象
    dataset = TensorDataset(x,y)

    #2.创建数据加载器对象
    #shuffle等于True表示每一个epoch之后就会打乱数据
    dataloader = DataLoader(dataset,batch_size=16,shuffle=True)

    #3.创建初始的线性回归模型
    #参数1是输入特征维度，参数2是输出特征维度
    model = nn.Linear(1, 1)
    """
    对于一个数据，列是特征维度，行是数据个数
    """

    #4.创建损失函数对象
    criterion = nn.MSELoss()

    #创建优化器对象
    #参数1：模型参数；参数2：学习率
    optimizer = optim.SGD(model.parameters(),lr=0.01)

    #process
    epochs, loss_list = 100, []
    for epoch in range(epochs):
        total_loss, total_examples = 0.0, 0
        #从数据加载器中获取批次数据
        for train_x, train_y in dataloader:#七批(16,16,16,16,16,16,4)
            y_pred = model(train_x)
            loss = criterion(y_pred, train_y.reshape(-1,1))#loss是这一批次的平均损失
            total_loss += loss.item()
            total_examples += 1#这个变量统计的是训练的总批次

            #梯度清零 + 反向传播 + 梯度更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        #统计本轮所有批次的平均损失
        loss_list.append(total_loss/total_examples)
        print(f"轮数：{epoch+1}.平均损失：{total_loss/total_examples:5f}")

    #打印结果
    print(f"{epochs}轮的平均损失分别为：{loss_list}")
    print(f"模型参数->权重：{model.weight},偏置：{model.bias}")

    #绘制损失曲线
    plt.plot(range(epochs),loss_list)
    plt.title("损失值曲线变化图")
    plt.grid()#绘制网格线
    plt.show()

    #绘制样本点分布情况
    plt.scatter(x,y)#散点图的意思
    plt.show()

    #绘制训练模型的预测值
    with torch.no_grad():
        y_pred = torch.tensor(data = [v * model.weight + model.bias for v in x])
    y_ture = x * coef + 14
    plt.plot(x,y_pred,color = 'red',label = "预测值")
    plt.plot(x,y_ture,color = 'green',label = "真实值")
    plt.legend()#显示图例
    plt.grid()
    plt.show()

x,y,coef = create_dataset()
train(x,y,coef)















