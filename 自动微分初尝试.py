import torch
"""
1.pytorch不支持向量张量对向量张量的求导，只支持标量张量对向量张量的求导，且要求数据都是float型
2.y.backward() y是一个标量
3.x.grad 获取x点的梯度（会自动累加）
"""
# #example1->演示基本算法
# w = torch.tensor(10,requires_grad=True,dtype=torch.float)
# loss = 2 * w ** 2
# print(f"梯度函数类型：{type(loss.grad_fn)}")
"""
MulBackward0 是 PyTorch 自动求导（Autograd）引擎中用于乘法操作的反向传播函数。
它出现在张量的 grad_fn 属性中，表示该张量是通过乘法运算得到的，并且这个乘法运算的反向求导逻辑由 MulBackward0 负责。
"""
# loss.backward() #计算梯度，计算完后会对应的梯度记录到w的grad属性中
# w.data = w.data - 0.01 * w.grad
# print(f'更新后权重w = {w.data}')
# print(type(w.grad))#注意梯度也是一个张量
# #更新100次
# for i in range(1, 101, 1):
#     print(f"第{i}次更新：")
#     loss = 2 * w ** 2
#     w.grad.zero_()#更新后必须清理梯度，不然会导致梯度累加
#     loss.backward()
#     w.data = w.data - 0.01 * w.grad
#     print(f'更新后权重w = {w.data:.5f}')#:.5f保留五位小数

# #example2 模拟预测函数y = x^2 + 0.6*x + 0.7
# #初始化权重
# w = torch.tensor([[0.5],[0.3],[0.4]],requires_grad=True,dtype=torch.float)
# print(f"初始权重为w = {w}")
# #给三个点作为输入
# x1 = 1.0
# x2 = 2.0
# x3 = -1.0
# x = torch.tensor([[1,1,1],
#                   [4,2,1],
#                   [1,-1,1]],dtype=torch.float)
# #三个真实值
# y1 = 2.3
# y2 = 5.9
# y3 = 1.1
# y = torch.tensor([[2.3],[5.9],[1.1]],dtype=torch.float)
"""
PyTorch 在计算梯度时要记住所有运算步骤。
比如 loss = (x @ w - y)^2 这个计算，PyTorch 会记下：“loss 是通过 -、^2 等操作从 w 算出来的”。这些信息形成一个计算图。
当你想修改 w 这个张量的值（比如更新权重），PyTorch 必须确保不会破坏它已经记下的历史。
如果你直接做 w.sub_(...)（原地修改），就等于擦掉 w 原来的值，PyTorch 再回头看历史时就会找不到原来的数据，导致无法正确计算梯度。
所以 PyTorch 干脆规定：禁止对需要梯度的叶节点进行原地修改，一旦你试图这么做，它就抛出 RuntimeError。
"""
# for i in range(1,501):
#     print(f"第{i}次跟新：")
#     if w.grad is not None:
#         w.grad.zero_()
#     loss = ((y - (x @ w))**2).mean()
#     loss.backward()
#     with torch.no_grad():#接下来的操作不计入计算图，即该步骤不参与梯度的计算
#         w.sub_(0.1 * w.grad)
#     print(f"跟新后w = {w}")
#     print(f"loss = {loss:.5f}")

























