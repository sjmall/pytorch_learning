"""
LSTM简介
core：利用sigmoid函数，tanh函数，和三个门来解决RNN短期记忆的问题
1.遗忘门：
    f_t = sigmoid(W_(xf) @ x_t + W_(hf) @ h_(t-1) + bf) 用来生成对老记忆h_(t-1)的保留比例

2.输入门：
    输入门分两个部分，一个用来生成记忆所有位置应该更新的比例，一个用来生成候选记忆
    i_t = sigmoid(W_(xi) @ x_t + W_(hi) @ h_(t-1) + bi) 用来生成记忆对应位置应该更新的比例
    c_t(bar) = tanh(W_(xc) @ x_t + W_(hc) @ h_(t-1) + bc) 用来生成候选记忆

细胞状态更新
    c_t = f_t · c_(t-1) + i_t · c_t(bar)

3.输出门：
    输出当前的短期记忆
    o_t = sigmoid(W_(xo) @ x + W_(h0) @ h_(t-1) + b/0)
    h_t = o_t · tanh(c_t)

API:
    nn.LSTM(
    input_size, 词向量的维度
    hidden_size, 隐藏状态的维度(c_t,h_t的长度)
    num_layers, 纵向层数
    batch_first)
    output, (h_t, c_t) = LSTM(x,(h_(t-1),c_(t-1))) 如果参数不传入(h,t)则会自动初始化为全0矩阵
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time

num_chars = 15
batch_size = 1024
lr = 0.001
epochs = 50
temperature = 1.2

#构建数据集
def build_vocab(txt_path='../data/Chinese_poem_data/Classical_Chinese_poetry_with_labels.txt'):
    unique_words, all_poems = [], []
    for line in open(txt_path, 'r', encoding='utf-8'):
        line = line.strip()
        if not line:
            continue

        parts = line.split(',')
        poem_text = parts[-1]
        words = list(poem_text)
        if not words:
            continue
        all_poems.append(words)
        for word in words:
            if word not in unique_words:
                unique_words.append(word)

    #构建字典
    word_to_index = {word:i for i, word in enumerate(unique_words)}
    word_count = len(unique_words)

    #将所有诗句文本转换为索引流
    corpus_idx = []
    for words in all_poems:
        tmp = [word_to_index[word] for word in words]
        corpus_idx.extend(tmp)

    return unique_words, word_to_index, word_count, corpus_idx

#构造数据加载器
class PoemDataset(Dataset):
    def __init__(self,corpus_idex,numchars):
        self.corpus_idex = corpus_idex
        self.num_chars = numchars
        self.All_num = len(self.corpus_idex)
        self.number = self.All_num - self.num_chars

    def __len__(self):
        return self.number

    def __getitem__(self, idx):
        start = min(max(0,idx),self.All_num-self.num_chars-1)
        end = start + self.num_chars
        x = self.corpus_idex[start:end]
        y = self.corpus_idex[start+1:end+1]
        return torch.tensor(x),torch.tensor(y)

#构建模型
class LSTMpoemGenerator(nn.Module):
    def __init__(self, word_count):
        super().__init__()
        #词嵌入层
        self.ebd = nn.Embedding(word_count, 128)

        self.lstm = nn.LSTM(128,256,1,batch_first=True)

        self.out = nn.Linear(256,word_count)
        self.out.weight = nn.init.xavier_normal_(self.out.weight)
        self.out.bias = nn.init.zeros_(self.out.bias)

    def forward(self, x, state):
        embd = self.ebd(x)
        output, state = self.lstm(embd, state)
        output = self.out(output.reshape(-1,256))
        return output, state

    def init_hidden(self, bs, device):
        h0 = torch.zeros(1, bs, 256, device=device)
        c0 = torch.zeros(1, bs, 256, device=device)
        return (h0, c0)

def train():
    #检查设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)

    #加载数据
    unq_words, word_to_index, word_count, corpus_idex = build_vocab()
    dataset = PoemDataset(corpus_idex,num_chars)
    poem_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    #初始化模型
    model = LSTMpoemGenerator(word_count).to(device)
    model.train()

    #训练
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print("start training...")
    for epoch in range(epochs):
        start_time = time.time()
        sam, total_loss = 0, 0.0

        for x,y in poem_loader:
            x, y = x.to(device), y.to(device)
            state = model.init_hidden(x.size(0), device)
            output, state = model(x, state)
            loss = criterion(output, y.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            sam += 1
        print(f"第{epoch+1}轮:耗时:{time.time()-start_time:.2f}s,损失:{total_loss/sam:.4f}")

    torch.save(model.state_dict(),'../model_params/LSTM_poem_generator.pth')
    print("train done!")

def evaluate(start_char, length):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)
    unique_words, word_to_index, word_count, corpus_idex = build_vocab()
    model = LSTMpoemGenerator(word_count).to(device)
    model.load_state_dict(torch.load('../model_params/LSTM_poem_generator.pth'))
    model.eval()
    if start_char not in word_to_index:
        print(f"No this words!")
        return
    state = model.init_hidden(1, device)
    word_idx = word_to_index[start_char]
    generate_sentence = [word_idx]
    x = torch.tensor([[word_idx]], dtype=torch.long, device=device)
    print("start predicting...")
    with torch.no_grad():
        for _ in range(length):
            output, state = model(x, state)
            logits = output/temperature
            probs = torch.softmax(logits, dim=-1)
            result_idx = torch.multinomial(probs, 1, replacement=True)
            generate_sentence.append(result_idx.item())
            x = torch.tensor([[result_idx.item()]], dtype=torch.long, device=device)

        for idx in generate_sentence:
            print(unique_words[idx],end='')
        print()
        print("over!")

evaluate("操",40)









