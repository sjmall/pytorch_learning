import torch
import torch.nn as nn
import torch.optim as optim
import jieba
from torch.utils.data import DataLoader
import time
import gradio as gr

num_chars = 15
batch_size = 80
lr = 0.001
epochs = 50
temperature = 1.3

# todo:获取数据
def bulid_vocab():
    # 记录去重后所有的词以及每行文本分词后的结果
    unique_words, all_words = [], []
    # 按行读取歌词，得到字符串类型的line
    for line in open('../data/Jay_lyrics_create_data/jaychou_lyrics.txt', 'r', encoding='utf-8'):
        words = jieba.lcut(line)
        all_words.append(words)

        for word in words:
            if word not in unique_words:
                unique_words.append(word)

    # 构建词表，字典形式，key是词，value是索引
    word_to_index = {word: i for i, word in enumerate(unique_words)}
    word_count = len(unique_words)

    # 将歌词文本有词表表示
    corpus_idx = []
    for words in all_words:
        tmp = []
        for word in words:
            tmp.append(word_to_index[word])
        corpus_idx.extend(tmp)

    return unique_words, word_to_index, word_count, corpus_idx

# todo:构造数据集
class LyricsDataset(torch.utils.data.Dataset):
    def __init__(self, corpus_idx, numchars):
        self.corpus_idx = corpus_idx
        self.num_chars = numchars
        self.word_count = len(self.corpus_idx)
        # 句子数量
        self.number = self.word_count - self.num_chars

    def __len__(self):
        return self.number

    def __getitem__(self, idx):
        start = min(max(0, idx), self.word_count - self.num_chars - 1)
        end = start + self.num_chars
        x = self.corpus_idx[start:end]
        y = self.corpus_idx[start + 1:end + 1]
        return torch.tensor(x), torch.tensor(y)

# todo:构造模型
class TextGenerator(nn.Module):
    def __init__(self, word_count):
        super().__init__()
        # 初始化词嵌入层
        self.ebd = nn.Embedding(word_count, 128)
        # 循环网络层
        self.rnn = nn.RNN(128, 256, 1, batch_first=True)
        # 全连接层
        self.out = nn.Linear(256, word_count)
        self.out.weight = nn.init.xavier_normal_(self.out.weight)
        self.out.bias = nn.init.zeros_(self.out.bias)

    def forward(self, x, h):
        # embedding会追加维度，将索引单独映射到一个新维度，原本x.shape from (batch_size, num_chars) to (batch_size, num_chars, 128)
        embd = self.ebd(x)
        output, h = self.rnn(embd, h)
        output = self.out(output.reshape(-1, 256))
        return output, h

    # 给 init_hidden 增加一个 device 参数
    def init_hidden(self, bs, device):
        # 将初始化的全零张量直接创建在指定的 device 上
        return torch.zeros(1, bs, 256, device=device)

# todo:训练
def train():
    # 检测并定义设备（优先使用GPU，如果没有则退回CPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前使用的计算设备为: {device}")

    unique_words, word_to_index, word_count, corpus_idx = bulid_vocab()
    dataset = LyricsDataset(corpus_idx, num_chars)

    # 将模型挂载到设备上
    model = TextGenerator(word_count).to(device)
    model.train()

    lyrics_loader = DataLoader(dataset, batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        start = time.time()
        sam, total_loss = 0, 0.0
        for x, y in lyrics_loader:
            # 将数据特征 x 和标签 y 送入设备
            x, y = x.to(device), y.to(device)

            # 初始化隐藏状态时传入 device
            h0 = model.init_hidden(x.size(0), device)

            output, hidden = model(x, h0)

            loss = criterion(output, y.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            sam += 1

        print(f"第{epoch + 1}轮, time:{time.time() - start:.2f}, loss:{total_loss / sam:.4f} ")

    torch.save(model.state_dict(), "../model_params/Jay_lyrics_generator.pth")

# todo:评估
def evaluate(start_word, length):
    # 1. 确定设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    unique_words, word_to_index, word_count, corpus_idx = bulid_vocab()

    # 2. 实例化模型并移动到对应设备
    model = TextGenerator(word_count).to(device)

    # 将模型权重加载到对应设备上 (修改 map_location)
    model.load_state_dict(torch.load("../model_params/Jay_lyrics_generator.pth", map_location=device))

    # 开启评估模式，关闭 Dropout/BatchNorm 等（好习惯）
    model.eval()

    # 初始化隐藏状态
    h0 = model.init_hidden(1, device)

    # 获取初始词的索引
    if start_word not in word_to_index:
        print(f"词表中不存在起始词: {start_word}")
        return

    word_idx = word_to_index[start_word]
    generate_sentence = [word_idx]

    # 将初始词转化为 Tensor，形状为 (1, 1) 即 (batch_size, seq_len)，并送入设备
    x = torch.tensor([[word_idx]], dtype=torch.long, device=device)

    # 3. 循环生成
    with torch.no_grad():  # 推理阶段不需要计算梯度，能大幅节省显存和加快速度
        for _ in range(length):
            output, h0 = model(x, h0)

            logits = output / temperature

            probs = torch.softmax(logits, dim=-1)

            # 按照概率分布进行随机采样
            # 参数1：一个一维或者二维张量表示概率分布 参数2：抽样的次数：参数3：抽样是否放回 该函数返回索引
            result_idx = torch.multinomial(probs, 1, replacement=False).item()

            generate_sentence.append(result_idx)
            x = torch.tensor([[result_idx]], dtype=torch.long, device=device)

    # 4. 打印结果
    for id in generate_sentence:
        print(unique_words[id], end='')
    print()  # 换行

# todo:创建网页
def web_evaluate(start_word, length, t):
    """专门给网页调用的生成函数"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    unique_words, word_to_index, word_count, corpus_idx = bulid_vocab()

    model = TextGenerator(word_count).to(device)
    model.load_state_dict(torch.load("../model_params/Jay_lyrics_generator.pth", map_location=device))
    model.eval()

    h0 = model.init_hidden(1, device)

    if start_word not in word_to_index:
        return f"兄弟，词表中没有【{start_word}】这个词，换个词试试？"

    word_idx = word_to_index[start_word]
    generate_sentence = [word_idx]
    x = torch.tensor([[word_idx]], dtype=torch.long, device=device)

    with torch.no_grad():
        for _ in range(int(length)):
            output, h0 = model(x, h0)
            logits = output / t
            probs = torch.softmax(logits, dim=1)
            result_idx = torch.multinomial(probs, num_samples=1).item()

            generate_sentence.append(result_idx)
            x = torch.tensor([[result_idx]], dtype=torch.long, device=device)

    # 不打印，而是组合成字符串返回
    lyric_result = "".join([unique_words[id] for id in generate_sentence])
    return lyric_result


# evaluate('爱情',111)

# # --- Gradio 界面配置 ---
# demo = gr.Interface(
#     fn=web_evaluate,  # 网页要绑定的 Python 函数
#     inputs=[
#         gr.Textbox(label="请输入起始词（比如：女人、晴天、退后）", value="女人"),
#         gr.Slider(minimum=20, maximum=200, step=10, value=100, label="生成歌词长度"),
#         gr.Slider(minimum=0.5, maximum=1.5, step=0.1, value=0.8, label="创造力温度 (越低越保守，越高越奔放)")
#     ],
#     outputs=gr.Textbox(label="AI方文山 倾情创作：", lines=8),
#     title="周杰伦歌词 AI 生成器",
#     description="基于 RNN 神经网络训练的歌词生成器，输入一个词，为你谱写周氏情歌。"
# )
#
# demo.launch(share = True)
