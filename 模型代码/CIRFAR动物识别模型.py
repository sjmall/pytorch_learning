import torch
import torch.nn as nn
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torchvision.transforms import ToTensor
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt
"""
CIFAR10是计算机视觉的自带数据集，内含六万多张图片(32,32,3)
分为五万张训练集和一万张测试集
共有十个类别，分别为0，1，2，3，4，5，6，7，8，9
"""
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")#使用GPU处理器

train_batch_size = 140
eval_batch_size = 256
n = 0.002
epoches = 70
dropout = 0.30
L2_weight_decay = 0

#todo:准备数据集
def create_dataset():
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),#填充4格，然后在40x40中随机裁剪出32
        transforms.RandomHorizontalFlip(),#随机翻转图片
        transforms.ToTensor()
    ])
    #获取训练集 (参三是数据是否转化为张量，参四是数据联网是否下载) (注意transform再转化为符合形状的张量，还回把0到255映射到0到1
    train_dataset = CIFAR10(root='../data/CIFAR10_data', train=True, transform=train_transform, download=True)
    #获取测试集
    test_dataset = CIFAR10(root='../data/CIFAR10_data', train=False, transform=ToTensor(), download=True)
    return train_dataset, test_dataset

def showdatasets():
    train_dataset, test_dataset = create_dataset()
    print(f"train_dataset's shape = {train_dataset.data.shape}")
    print(f"test_dataset's shape = {test_dataset.data.shape}")
    print(f"class of data: {train_dataset.class_to_idx}")

    plt.imshow(train_dataset.data[11])
    plt.title(train_dataset.targets[11])
    plt.show()

#todo:构建网络
class ImageModel(nn.Module):
    def __init__(self):
        super().__init__()
        #搭建网络结构
        self.dropout = nn.Dropout(dropout)
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.norm1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.norm2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.norm3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, stride=2, padding=0)

        self.linear1 = nn.Linear(128 * 8 * 8, 256)
        nn.init.kaiming_normal_(self.linear1.weight)
        nn.init.zeros_(self.linear1.bias)

        self.linear2 = nn.Linear(256, 512)
        nn.init.kaiming_normal_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)

        self.linear3 = nn.Linear(512, 128)
        nn.init.kaiming_normal_(self.linear3.weight)
        nn.init.zeros_(self.linear3.bias)

        self.output = nn.Linear(128, 10)
        nn.init.xavier_normal_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x):
        x = torch.relu(self.norm1(self.conv1(x)))
        x = self.pool(x)
        x = torch.relu(self.norm2(self.conv2(x)))
        x = self.pool(x)
        x = torch.relu(self.norm3(self.conv3(x)))
        #现在x's shape is (train_batch_size,128,8,8)
        x = x.reshape(x.size(0), -1)

        x = torch.relu(self.linear1(x))
        x = self.dropout(x)
        x = torch.relu(self.linear2(x))
        x = self.dropout(x)
        x = torch.relu(self.linear3(x))
        x = self.dropout(x)
        return self.output(x)

#todo:train
def train_model(train_dataset):
    dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    model = ImageModel().to(device)
    model.train()
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=n, weight_decay= L2_weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.2)
    for epoch in range(epoches):
        sample, correct, total_loss = 0, 0, 0.0
        start = time.time()
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)#将数据加载在GPU上
            y_pred = model(x)
            loss = criterion(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (y_pred.argmax(dim=-1) == y).sum().item()
            sample += len(y)
        scheduler.step()

        print(f"第{epoch+1}轮用时: {time.time() - start:.3f}s")
        print(f"第{epoch+1}轮损失: {total_loss / sample:.5f}")
        print(f"第{epoch+1}轮准确率: {correct / sample * 100:.2f}%")
        print('-' * 60)

    torch.save(model.state_dict(), '../model_params/img_model.pth')

def evaluate_model(test_dataset):
    dataloader = DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=False)
    model = ImageModel().to(device)
    model.load_state_dict(torch.load('../model_params/img_model.pth', map_location='cpu'))#先把参数加载到CPU上确保能被读取
    #CPU上的能被GPU读取，但是GPU上的不能直接被CPU读取
    model.eval()
    correct, total = 0, 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        y_pred = model(x)
        y_pred = y_pred.argmax(dim=-1)
        correct += (y_pred == y).sum().item()
        total += len(y)

    print(f"测试总正确率: {correct/total*100:.2f}%")


traindataset, testdataset = create_dataset()
# train_model(traindataset)
evaluate_model(testdataset)



































