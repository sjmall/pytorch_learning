"""
循环网络层的处理步骤
    上一次隐藏状态 + 这一次输入 -> 这一次隐藏状态 -> 本次输出

公式:
    # 隐藏状态更新
    h_t = tanh( W_hh @ h_{t-1} + W_xh @ x_t + b_h )

    # 输出计算（通常用线性层，不用tanh）交给全连接层
    y_t = W_hy @ h_t + b_y

输出结果的每个位置代表当前词是词表中对应词的得分

API:
    RNN = torch.nn.RNN(input_size, hihdden_size, num_layers) 分别代表输入数据的维度，隐藏层的维度，隐藏层的层数
    rephrase:输入数据的维度就是词向量的维度，隐藏层的维数就是h_t的长度，隐藏层的层数就是一个一个输入要经过多少层RNN后输出
API_using:

"""
import torch
import torch.nn as nn

#为了计算方便这里莫名其妙的把batchsize的参数位置放在了第二个位置上
rnn = nn.RNN(input_size=128, hidden_size=256, num_layers=1, batch_first=False)

#位置一代表一个句子有几个词，位置二代表句子的数量，位置三代表词向量的维度
x = torch.randn(size=(5, 32, 128))

#位置一代表隐藏层的层数，位置二代表句子的数量，位置三代表隐藏层向量的维度
h0 = torch.randn(size=(1, 32, 256))

"""
output是所有时间步的 256 维特征的大合集（有5个）。
output可以用来逐词翻译等，在这里能取到每一个时间步的隐藏状态
h1仅仅是最后一个时间步的 256 维特征（只有1个）。
h1用来判断结果如句子的褒贬，在这里只需要最后一步整合了整个句子信息的h1隐藏状态即可
"""
output, h1 = rnn(x, h0)
print(f"output: {output.shape}")
print(f"h0: {h0.shape}")