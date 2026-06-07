"""
批量归一化（Batch Normalization）
================================

【思想】
    每一层输入的分布会在训练过程中不断变化（Internal Covariate Shift，
    内部协变量偏移），这会导致：
    1. 网络需要不断适应新的分布，收敛变慢
    2. 深层网络训练困难（梯度消失/爆炸）

    批量归一化的核心思想是：
    对每一层的输入进行标准化，使其均值接近 0、方差接近 1，
    让数据分布保持稳定，从而加速训练并提高稳定性。

【公式】
    给定一个 batch 的输入（一维特征） x = [x₁,
                                       x₂,
                                       ...,
                                       xₘ]（m 为 batch size）：

    1. 计算 batch 均值：      μ_B = (1/m) * Σxᵢ
    2. 计算 batch 方差：      σ²_B = (1/m) * Σ(xᵢ - μ_B)²
    3. 标准化：               x̂ᵢ = (xᵢ - μ_B) / √(σ²_B + ε)
                                （ε 是一个很小的数，防止除零）
    4. 缩放和平移（可学习）：   yᵢ = γ * x̂ᵢ + β

    其中 γ（缩放参数，初始为 1）和 β（偏移参数，初始为 0）是可学习的参数，
    让网络自己决定标准化的"强度"——如果不希望标准化，也可以学回原始分布。

【目的】
    1. 加速收敛：梯度更稳定，可以使用更大的学习率
    2. 缓解梯度消失/爆炸：把数据拉回激活函数的敏感区
    3. 提供一定的正则化效果（有轻微的正则化作用，可减少 Dropout 的需求）
    4. 减少对参数初始化的敏感度：网络更容易调参

【做法】
    PyTorch 中直接使用 nn.BatchNorm1d / nn.BatchNorm2d 即可：

        nn.BatchNorm1d(num_features)   # 用于全连接层（1D）
        nn.BatchNorm2d(num_features)   # 用于卷积层（2D）
        1d表示每个特征是一维的，一般传入(a,b) a表示一批次的数量，b表示一个样本有多少特征
        2d表示每个特征是二维的，一般传入(m,n,a,b) m表示一批次样本数，n表示特征数，(a,b)表示一个特征的形状
        注意归一化是以一个批次所有样本的同一特征值做归一化，意味着有多少特征就有多少个γ，β

    通常放在 线性层/卷积层 之后、激活函数 之前：

        x = self.fc1(x)
        x = self.bn1(x)     # BN 在激活之前
        x = torch.relu(x)

    注意：训练时用 net.train()，测试时用 net.eval()
    - 训练时：使用当前 batch 的均值和方差
    - 测试时：使用训练集累积的全局均值和方差（running mean / running var）

【与正则化.py 中三种正则化的对比】
    | 方法         | 核心作用           | 训练/测试差异               |
    |-------------|-------------------|----------------------------|
    | L1 正则化    | 稀疏解，特征选择     | 无差异                     |
    | L2 正则化    | 权重衰减，防止过拟合  | 无差异                     |
    | Dropout     | 随机丢弃，集成学习   | 训练时丢弃，测试时关闭      |
    | BatchNorm   | 标准化分布，加速训练  | 训练用 batch 统计，测试用全局统计 |

"""
import torch
import torch.nn as nn
def eg1():
    input_2d = torch.randn(size = (2,2,3,4))
    print(f"input_2d = {input_2d}")
    print('-' * 50)
    #todo:创建归一化对象
    #这里的动量公式与梯度更新的不一样：s_t(mean) = (1-momenntum) * s_{t-1}(mean) + momentum * current(mean) mean可替换成var
    #affine表示是否添加可学习的参数β，γ（每个特征有属于自己的βi，γi，且二者都是一维张量，β初始化为0，gamma初始化为1）
    bn2d = nn.BatchNorm2d(num_features=2, eps=1e-05, momentum=0.1, affine=True)
    output_2d = bn2d(input_2d)
    print(f"output_2d = {output_2d}")

def eg2():
    torch.manual_seed(42)
    input_1d = torch.randn(size = (3,5))
    print(f"input_1d = {input_1d}")
    print('-' * 50)
    l = nn.Linear(5,2)
    mid_1d = l(input_1d)
    print(f"mid_1d = {mid_1d}")
    print('-' * 50)
    bn1d = nn.BatchNorm1d(num_features=2, eps=1e-05, momentum=0.1, affine=True)
    output_1d = bn1d(mid_1d)
    print(f"output_1d = {output_1d}")

eg2()



