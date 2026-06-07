"""
1.指数加权平均
    递推式（递归更新形式）：
    v_t = β * v_{t-1} + (1-β) * θ_t
    展开式（指数衰减形式）：
    v_t = (1-β)θ_t + β(1-β)θ_{t-1} + β²(1-β)θ_{t-2} + β³(1-β)θ_{t-3} + …

    特别的，β越小说明加权平均越侧重于当前梯度，反之
    用v_t代替梯度来更新，这是Momentum法

2.累计平方梯度
    递推式：s_t = s_{t-1} + g_t * g_t
    学习率：n = n/(sqrt(s_t)+o) o为小常数：1e-10
    此为自适应学习率Adagrad法，虽然能是模型更好的收敛到极小值，但是也容易导致学习率过早的收敛导致逃不出鞍点

3.指数加权累计平方梯度
    递推式：s_t = β * s_{t-1} + (1-β) * g_t * g_t
    学习率公式不变 n = n/(sqrt(s_t)+o)
    此为自适应学习率RMSprop法，自带遗忘机制，可以在梯度大的区域逐渐降低学习率，梯度小的时候逐渐升高学习率

4.Adam RMSprop + Momentum
    结合 Momentum（一阶动量）和 RMSprop（二阶动量）的思想

    一阶动量（梯度的指数加权平均）：
        m_t = β₁ * m_{t-1} + (1-β₁) * g_t

    二阶动量（梯度平方的指数加权平均）：
        v_t = β₂ * v_{t-1} + (1-β₂) * g_t * g_t

    偏差修正（解决初始时刻估计偏向于 0 的问题：m_t和v_t初始化为0向量，在t较小时会严重偏向0，
    尤其当β接近1时衰减慢，偏差更显著。除以(1-β^t)可放大估计值到无偏水平，t增大后修正逐渐消失）：
        m_t_hat = m_t / (1 - β₁^t)
        v_t_hat = v_t / (1 - β₂^t)

    参数更新：
        θ_t = θ_{t-1} - η * m_t_hat / (sqrt(v_t_hat) + ε)

    其中：
        β₁: 一阶动量衰减系数，通常取 0.9
        β₂: 二阶动量衰减系数，通常取 0.999
        ε: 小常数，防止除零，通常取 1e-8
        η: 学习率
        g_t: t 时刻的梯度
        θ_t: t 时刻的参数
"""
import torch
import torch.optim as optim

def momentum_sgd():
    w = torch.tensor([1.0],requires_grad=True)
    optimizer = optim.SGD([w], lr=0.1, momentum=0.1)  # 参数1：待优化的参数列表；lr：学习率；momentum：动量参数β
    for _ in range(100):
        loss = ((w ** 2)/2.0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"w:{w.item():4f}, loss:{loss.item():4f}")

def adagrad():
    w = torch.tensor([1.0],requires_grad=True)
    optimizer = optim.Adagrad([w], lr=0.01)
    for _ in range(100):
        loss = ((w ** 2)/2.0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"w:{w.item():4f}, loss:{loss.item():4f}")

def rmsprop():
    w = torch.tensor([1.0],requires_grad=True)
    optimizer = optim.RMSprop([w], lr=0.01, alpha=0.9)
    for _ in range(100):
        loss = ((w ** 2)/2.0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"w:{w.item():4f}, loss:{loss.item():4f}")

def adam():
    w = torch.tensor([1.0],requires_grad=True)
    optimizer = optim.Adam([w], lr=0.1, betas=(0.9, 0.999), eps=1e-8)
    for _ in range(100):
        loss = ((w ** 2)/2.0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"w:{w.item():4f}, loss:{loss.item():4f}")

adam()
