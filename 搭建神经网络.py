"""
流程
1.定义一个类，继承nn.Module
2.在_init_()方法中，搭建神经网络
3.在forward()方法中，完成前向传播
"""
import torch.nn as nn
import torch
import torchsummary as s #计算模型参数，查看模型结构
class ModelDemo(nn.Module):
    #todo:在init魔法方法中完成初始化：父类成员，神经网络搭建
    def __init__(self):
        #1.初始化父类成员
        super().__init__()
        #2.搭建神经网络 注意不用考虑输入层
        self.linear1 = nn.Linear(3,3)
        self.linear2 = nn.Linear(3,2)
        self.output = nn.Linear(2,2)
        nn.init.xavier_normal_(self.linear1.weight)
        nn.init.zeros_(self.linear1.bias)
        nn.init.xavier_normal_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)
        
    #todo:前向传播
    def forward(self,x):
        x = self.linear1(x)
        x = torch.tanh(x)
        x = self.linear2(x)
        x = torch.tanh(x)
        x = self.output(x)
        x = torch.softmax(x,dim=-1)#在列变化的维度处理（在这里即按行）
        return x
def train():
    model = ModelDemo()
    print(model)
    print('-' * 50)
    data = torch.randn((5,3))
    print(data)
    output = model(data)
    print(output)
    print('-' * 50)

    #计算，查看模型参数
    s.summary(model,input_size = (5,3))
    for name, param in model.named_parameters():
        print(f"name: {name}")
        print(f"param: {param}\n")

train()