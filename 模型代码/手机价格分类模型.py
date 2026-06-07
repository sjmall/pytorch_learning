"""
==============================
Pandas DataFrame 笔记
==============================

data = pd.read_csv('train.csv')  →  data 是 pandas.DataFrame 对象

【类型】: pandas.DataFrame（二维表格数据结构）
【不是张量】: 不是 torch.Tensor，也不是 numpy.ndarray

【特点】
  - 有行索引（index）和列索引（columns，即表头）
  - 每列可以是不同的数据类型（int, float, str, bool...）
  - 类似 Excel 表格，支持丰富的表格操作

【常用功能】
  data.head()          → 查看前5行
  data.info()          → 查看列名、非空数、数据类型
  data.describe()      → 数值列的统计摘要（均值、标准差等）
  data.columns         → 获取所有列名
  data.shape           → (行数, 列数)
  data['col_name']     → 取单列（返回 Series）
  data[['c1','c2']]    → 取多列（返回 DataFrame）
  data.iloc[i, j]      → 按位置索引
  data.loc[row_lbl, col_lbl] → 按标签索引
  data.dropna()        → 删除含空值的行
  data.fillna(val)     → 填充空值

【转换为张量的标准流程】
  1. X = data.drop('label', axis=1).values     → numpy array
     y = data['label'].values
  2. X_tensor = torch.tensor(X, dtype=torch.float32)
     y_tensor = torch.tensor(y, dtype=torch.long)
  或一步: torch.tensor(data.to_numpy(), dtype=torch.float32)

【其他读取方式】
  pd.read_excel('file.xlsx')
  pd.read_json('file.json')
  pd.read_sql(sql, connection)
==============================
"""

#手机价格分类
#todo：根据手机的二十个特征映射成四个价格区间
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd #用来专门处理csv，excel中的数据
import time #时间模块

epochs = 200
batchsize = 32
lr = 0.005
dropout = 0.5
betas = (0.9, 0.999)

def create_dataset():
    # 加载csv文件数据集
    data = pd.read_csv('../data/phoneprice_data/train.csv')
    # print(data.columns)
    # print(data.shape)

    #获取x特征列，y标签列
    x, y = data.iloc[:, :-1], data.iloc[:, -1]
    # print(x.shape, y.shape) # (2000,20),(2000,)

    #将特征数据都转换为浮点型
    x = x.astype(np.float32)

    #切分训练集和测试集
    """
    参数1，2：带划分的特征数据和目标数据
    参数3：测试集所占比例
    参数4：随机种子
    参数5：依据y中的标签比例划分（比如A；B = 6：4，则划分后测试集与训练集中的比例都是6：4
    """
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
    # 标准化
    """
    创建了一个标准化对象，内置mean，std，一开始为空，若要对数据进行标准化，需要先学习到mean，std
    第一步是学习x_train中的mean，std，并且用来对x_train标准化，帮助训练
    第二步是用学习到的mean，std对x_test标准化
    注意不用x_test自己的mean，std标准化是因为测试过程要当作从未见到test的内容，真实情况我们不会手机需要预测的数据然后标准化
    """
    #注意转化后变成numpy数组
    transfer = StandardScaler()#返回的是numpy数组
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    train_dataset = TensorDataset(torch.from_numpy(x_train), torch.tensor(y_train.values))#.values返回numpy数组,但是是不可写的，浅拷贝会发出警告
    #特别的，返回的列数据会自动退化成一维数据
    test_dataset = TensorDataset(torch.from_numpy(x_test), torch.tensor(y_test.values))

    return train_dataset, test_dataset


class PhonePriceModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 定义层
        self.linear1 = nn.Linear(input_dim, 128)
        self.norm1 = nn.LayerNorm(128)  #层标准化，内含γ，β可训练

        self.linear2 = nn.Linear(128, 256)
        self.norm2 = nn.LayerNorm(256)  

        self.linear3 = nn.Linear(256, 512)
        self.norm3 = nn.LayerNorm(512)  

        self.linear4 = nn.Linear(512, 128)
        self.norm4 = nn.LayerNorm(128)  

        self.output = nn.Linear(128, output_dim)

    def forward(self, x):
        # 推荐结构：Linear -> Norm -> Activation -> Dropout
        x = self.dropout(torch.nn.LeakyReLU(0.1)(self.norm1(self.linear1(x))))
        x = self.dropout(torch.nn.LeakyReLU(0.1)(self.norm2(self.linear2(x))))
        x = self.dropout(torch.nn.LeakyReLU(0.1)(self.norm3(self.linear3(x))))
        x = self.dropout(torch.nn.LeakyReLU(0.1)(self.norm4(self.linear4(x))))
        x = self.output(x)
        return x

def train(train_dataset,input_dim,output_dim):
    train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True)
    model = PhonePriceModel(input_dim, output_dim)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, betas = betas, eps=1e-08)
    start = time.time()
    for epoch in range(epochs):
        total_loss = 0.0
        batch_num = 0
        for x, y in train_loader:#这里取出来的x，y是一个批次里所有的数据
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batch_num += 1
        print(f"epoch {epoch+1}, loss: {total_loss/batch_num}, time: {time.time()-start:.2f}s")

    torch.save(model.state_dict(), '../model_params/phoneprice.pth')
    print(model.state_dict())

def evaluate(test_dataset,input_dim,output_dim):
    model = PhonePriceModel(input_dim, output_dim)
    model.load_state_dict(torch.load('../model_params/phoneprice.pth'))
    model.eval()
    test_loader = DataLoader(test_dataset, batch_size=20)
    correct = 0
    for x, y in test_loader:
        correct_now = 0
        y_pred = model(x)
        y_pred = torch.argmax(y_pred, 1)#argmax获取最大元素对应的下标，即为类别
        print(f"y_pred: {y_pred}\ny_true: {y}")
        print(y_pred == y)
        correct += (y_pred == y).sum()
        correct_now += (y_pred == y).sum()
        print(f"当前批次准确率 = {correct_now/20 * 100:.2f}%")
        print()

    print(f"准确率 = {correct/len(test_dataset) * 100:.2f}%")

train_dataset, test_dataset = create_dataset()
# train(train_dataset,20,4)
evaluate(test_dataset,20,4)