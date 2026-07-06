import torch
import copy
from torch import nn
from torch.utils.data import Dataset, DataLoader
import math
import torch.optim as optim

#todo:定义transformer翻译的四个特殊标记
PAD_IDX, SOS_IDX, EOS_IDX, UNK_IDX = 0, 1, 2, 3
PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN = '<PAD>', '<SOS', '<EOS>', '<UNK>'

#todo:构建词表
def build_vocab(text_path = '../data/translation_data/Chinese_to_English.txt'):
    #todo:初始化词表中的特殊标记
    en_words = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]
    ch_words = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]
    pairs = []

    #todo:读取txt
    current_pair = {}
    for line in open(text_path,'r',encoding='utf-8'):
        line = line.strip()
        if not line: continue
        if line == '--':#此时代表我们处理好了上一段数据
            if 'english' in current_pair and 'mandarin' in current_pair:
                en_sent = current_pair['english'].lower().replace('.',' .').replace('?',' ?').replace('!',' !').replace(',',' ,')
                ch_sent = current_pair['mandarin']

                en_tokens = en_sent.split()#review:去除所有的空格并返回列表
                ch_tokens = list(ch_sent)

                pairs.append((en_tokens, ch_tokens))

                #todo:补充词表
                for w in en_tokens:
                    if w not in en_words: en_words.append(w)
                for w in ch_tokens:
                    if w not in ch_words: ch_words.append(w)

            current_pair = {}

        elif ':' in line:
            parts = line.split(':',1)
            key = parts[0].strip().lower()
            val = parts[1].strip()
            current_pair[key] = val

    #todo:构建字典
    en_w2i = {w: i for i, w in enumerate(en_words)}
    ch_w2i = {w: i for i, w in enumerate(ch_words)}

    return en_words, en_w2i, ch_words, ch_w2i, pairs

#todo:构建数据加载器
class TranslationDataset(Dataset):
    def __init__(self, pairs, en_w2i, ch_w2i, max_len=30):
        self.pairs = pairs
        self.en_w2i = en_w2i
        self.ch_w2i = ch_w2i
        self.max_len = max_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        en_tokens, ch_tokens = self.pairs[idx]

        en_ids = [self.en_w2i.get(w, UNK_IDX) for w in en_tokens] + [EOS_IDX]
        ch_ids = [SOS_IDX] + [self.ch_w2i.get(w, UNK_IDX) for w in ch_tokens] + [EOS_IDX]

        #todo:限制最大长度+PAD填充
        en_ids = en_ids[:self.max_len] + [PAD_IDX] * max(0, (self.max_len - len(en_ids)))
        ch_ids = ch_ids[:self.max_len+1] + [PAD_IDX] * max(0, ((self.max_len+1) - len(ch_ids)))

        #todo:转换为张量
        src = torch.tensor(en_ids, dtype=torch.long)#英文输入
        tgt = torch.tensor(ch_ids, dtype=torch.long)#中文输入

        return src, tgt

#todo:构建位置编码层
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        #todo:创建位置矩阵，注意d_model必须是偶数
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        #偶数用sin，奇数用cos
        """
        position's shape is [100,1], div_term shape is [d_model/2]
        利用广播机制相乘
        """
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        #todo:增加batch的维度
        pe = pe.unsqueeze(0)

        #todo:将pe与模型参数绑定，到时候能一起转到gpu，且不参与参数更新
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

#todo:构建注意力层
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.d_model = d_model
        #todo:确保词向量维度能被头整除
        assert d_model % n_head == 0

        self.d_k = d_model // n_head

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.fc = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        #todo:线性变换并切分成多头形状 [batch, seq_len, n_head, d_k] -> [batch, n_head, seq_len, d_k]
        Q = self.w_q(q).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)
        K = self.w_k(k).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)
        V = self.w_v(v).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)

        #todo:计算注意力得分 注意高维矩阵乘法只对最低两个维度做乘法，且要求高维度完全相同
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)#mask是一个矩阵形状与scores相同，为0的地方就把scores对应的地方换成-1e9

        attn_weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, V)

        #contiguous就是把内存按照索引连续的复制一份
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        return self.fc(context)

#todo:构建前馈神经网络
class Attention(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        #todo:残差链接 + LayerNorm
        residual = x
        x = self.w_2(self.dropout(torch.relu(self.w_1(x))))
        return self.layer_norm(x + residual)

#todo:构建encoder层
class EncoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, feed_forward, dropout=0.1):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        residual = x
        x = self.dropout(self.self_attn(x, x, x, mask))
        x = self.sublayer_norm(x + residual)

        x = self.feed_forward(x)
        return x

#todo:创建encoder器
class Encoder(nn.Module):
    def __init__(self, encoder_layer, N):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(N)])
        self.norm = nn.LayerNorm(encoder_layer.sublayer_norm.normalized_shape)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

#todo:构建decoder层
class DecoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, src_attn, feed_forward, dropout=0.1):
        super().__init__()
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, src_mask, tgt_mask):
        residual = x
        x = self.norm1(x)
        x = residual + self.dropout(self.self_attn(x, x, x, tgt_mask))

        residual = x
        x = self.norm2(x)
        x = residual + self.dropout(self.src_attn(x, k=memory, v=memory, mask=src_mask))

        residual = x
        x = self.norm3(x)
        x = residual + self.dropout(self.feed_forward(x))

        return x

#todo:encoder器
class Decoder(nn.Module):
    def __init__(self, decoder_layer, N):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(N)])
        self.norm = nn.LayerNorm(decoder_layer.norm1.normalized_shape)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)

        return self.norm(x)

#todo:构建transformer合适的优化器
def get_std_opt(model, d_model, warmup_steps = 4000):
    optimizer = optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-09)

    lr_lambda = lambda step: (d_model ** -0.5) * min((step + 1) ** -0.5, (step+1) * (warmup_steps ** -1.5))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    return optimizer, scheduler

#todo:构建完整transformer模型
class Transformer(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        memory = self.encode(src, src_mask)
        return self.decode(memory, src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.generator(self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask))

#todo:构建batch类
class Batch:
    def __init__(self, src, tgt, pad_idx=0):
        self.src = src
        self.src_mask = (src != pad_idx).unsqueeze(-2)

        if tgt is not None:
            self.tgt = tgt[:, :-1]
            self.tgt_y = tgt[:, 1:]

            self.tgt_mask = self.make_std_mask(self.tgt, pad_idx)

            self.ntokens = (self.tgt_y != pad_idx).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        tgt_mask = (tgt != pad).unsqueeze(-2)
        sz = tgt.size(-1)
        subsequent_mask = torch.triu(torch.ones((1, sz, sz), dtype=torch.uint8), diagonal=1) == 0
        subsequent_mask = subsequent_mask.to(tgt.device)
        return tgt_mask & subsequent_mask

#todo:构建一个epoch的过程
def train_epoch(model, data_loader, optimizer, scheduler, criterion, device):
    model.train()
    total_loss = 0
    total_tokens = 0

    for i, (src_data, tgt_data) in enumerate(data_loader):
        batch = Batch(src_data, tgt_data, pad_idx=PAD_IDX)

        src = batch.src.to(device)
        tgt = batch.tgt.to(device)
        tgt_y = batch.tgt_y.to(device)
        src_mask = batch.src_mask.to(device)
        tgt_mask = batch.tgt_mask.to(device)

        optimizer.zero_grad()

        out = model(src, tgt, src_mask, tgt_mask)

        loss = criterion(out.contiguous().view(-1, out.size(-1)), tgt_y.contiguous().view(-1))

        loss.backward()

        optimizer.step()
        scheduler.step()

        total_loss += loss.item() * batch.ntokens
        total_tokens += batch.ntokens

        if i % 100 == 0:
            print(f"--> Batch {i:04d} | Current Batch Loss: {loss.item():.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

    return total_loss / total_tokens

#todo:测试代码
def translate(model, english_sentence, en_w2i, ch_words, device, max_len=100):
    """
    针对输入的一句英文，使用训练好的模型进行自回归（逐字）翻译
    """
    model.eval()

    with torch.no_grad():
        en_sent = english_sentence.lower().replace('.', ' .').replace('?', ' ?').replace('!', ' !').replace(',', ' ,')
        en_tokens = en_sent.split()

        en_ids = [en_w2i.get(w, UNK_IDX) for w in en_tokens] + [EOS_IDX]

        src = torch.tensor(en_ids, dtype=torch.long).unsqueeze(0).to(device)
        src_mask = (src != PAD_IDX).unsqueeze(-2).to(device)

        memory = model.encode(src, src_mask)

        ys = torch.ones(1, 1).fill_(SOS_IDX).type(torch.long).to(device)

        for i in range(max_len):
            tgt_mask = Batch.make_std_mask(ys, PAD_IDX).to(device)

            out = model.decode(memory, src_mask, ys, tgt_mask)

            prob = out[:, -1]

            _, next_word = torch.max(prob, dim=-1)
            next_word = next_word.item()

            ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)

            if next_word == EOS_IDX:
                break

        ch_ids = ys.squeeze(0).tolist()
        translation_result = []
        for idx in ch_ids:
            #过滤掉特殊的启动符
            if idx in [SOS_IDX, EOS_IDX, PAD_IDX]:
                continue
            translation_result.append(ch_words[idx])

        return "".join(translation_result)

if __name__ == '__main__':
    #todo:超参数定义
    d_model = 768  # 词向量维度
    n_head = 12  # 多头注意力的头数
    d_ff = 3072 # 前馈神经网络中间层维度
    N = 8
    batch_size = 64
    epochs = 130

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    #todo:构建词表
    en_words, en_w2i, ch_words, ch_w2i, pairs = build_vocab()

    src_vocab_size = len(en_words)
    tgt_vocab_size = len(ch_words)
    print(f"英文词表大小: {src_vocab_size} | 中文词表大小: {tgt_vocab_size}")

    #todo:实例化数据集和加载器
    dataset = TranslationDataset(pairs, en_w2i, ch_w2i, max_len=30)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    #todo:实例化各个核心组件
    self_attn = MultiHeadAttention(d_model, n_head)
    src_attn = MultiHeadAttention(d_model, n_head)

    ffn = Attention(d_model, d_ff)

    encoder_layer = EncoderLayer(d_model, copy.deepcopy(self_attn), copy.deepcopy(ffn))
    encoder = Encoder(encoder_layer, N)

    decoder_layer = DecoderLayer(d_model, copy.deepcopy(self_attn), copy.deepcopy(src_attn), copy.deepcopy(ffn))
    decoder = Decoder(decoder_layer, N)

    src_embed = nn.Sequential(
        nn.Embedding(src_vocab_size, d_model, padding_idx=PAD_IDX),
        PositionalEncoding(d_model)
    )
    tgt_embed = nn.Sequential(
        nn.Embedding(tgt_vocab_size, d_model, padding_idx=PAD_IDX),
        PositionalEncoding(d_model)
    )

    generator = nn.Linear(d_model, tgt_vocab_size)

    model = Transformer(encoder, decoder, src_embed, tgt_embed, generator)
    model.to(device)

    choice = '1'
    while choice == "1" or choice == "2":
        print("Please enter your choice: 2[test] | 1[train] | 0[exit]")
        choice = input()
        if choice == "1":
            #todo:实例化优化器、学习率调度、损失函数
            optimizer, scheduler = get_std_opt(model, d_model=d_model)

            criterion = nn.CrossEntropyLoss(label_smoothing=0.1, ignore_index=PAD_IDX)

            #todo:训练
            print("begin to train...")
            best_loss = float('inf')  #记录最好成绩

            for epoch in range(epochs):
                epoch_loss = train_epoch(model, train_loader, optimizer, scheduler, criterion, device)
                print(f"Epoch [{epoch + 1:02d}/{epochs:02d}] | Avg Loss: {epoch_loss:.4f}")

                torch.save(model.state_dict(), "../model_params/transformer_translation/transformer_latest.pt")

                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    torch.save(model.state_dict(), "../model_params/transformer_translation/transformer_best.pt")
                    print("A best model has been saved!")

            print("END!")
        elif choice == "2":
            #todo:开始测试
            print("\n" + "=" * 40)
            print("Begin to translate...")

            test_model = Transformer(encoder, decoder, src_embed, tgt_embed, generator)

            best_weight_path = "../model_params/transformer_translation/transformer_best.pt"
            test_model.load_state_dict(torch.load(best_weight_path, map_location=device))
            test_model.to(device)
            print("The model has been loaded!")
            print("=" * 40 + "\n")

            test_sentence = input("Please enter your sentence: ")
            res = translate(test_model, test_sentence, en_w2i, ch_words, device)
            print(f"Original English sentence: {test_sentence}")
            print(f"Chinese sentence translated by model: {res}")
            print("-" * 30)





















