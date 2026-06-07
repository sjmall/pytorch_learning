import torch
import matplotlib.pyplot as plt
import torch.optim as optim

# 解决 matplotlib 中文显示问题
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 解决 matplotlib 中文显示问题
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def draw(time, n, ax):
    w = torch.tensor([3], dtype=torch.float, requires_grad=True)
    w_list = []
    optimizer = optim.SGD([w], lr=n)
    for _ in range(time):
        loss = w ** 2
        w_list.append(w.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 绘制损失函数图像
    x = torch.linspace(-5, 5, 100)
    y = x ** 2

    ax.plot(x.numpy(), y.numpy(), label='y = x²', color='blue')

    # 在图像上标记 w_list 中的点
    w_values = torch.tensor(w_list)
    loss_values = w_values ** 2
    ax.scatter(w_values.numpy(), loss_values.numpy(),#scatter绘制散点
                color='red', s=40, zorder=5, label=f'SGD 收敛路径 (lr={n})')
    # 用虚线连接点的顺序，展示收敛轨迹
    ax.plot(w_values.numpy(), loss_values.numpy(),
             color='red', linestyle='--', linewidth=1, alpha=0.6)

    # 标注起点和终点
    ax.scatter(w_list[0], w_list[0] ** 2, color='green', s=80, zorder=6,
                marker='o', label='起点')
    ax.scatter(w_list[-1], w_list[-1] ** 2, color='purple', s=80, zorder=6,
                marker='*', label='终点')

    ax.set_xlabel('w')
    ax.set_ylabel('Loss')
    ax.set_title(f'SGD 优化 y = x² (学习率={n}, 迭代次数={time})')
    ax.legend()
    ax.grid(True, alpha=0.3)
if __name__ == '__main__':
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    draw(10, 0.01, axes[0])
    draw(10, 0.1, axes[1])
    draw(10, 1.1, axes[2])

    plt.tight_layout()
    plt.show()
