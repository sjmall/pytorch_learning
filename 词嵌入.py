import torch
import jieba #结巴分词器
import torch.nn as nn

#场景一->演示如何把一个词的索引变为词向量
def dm():
    text = "门突然开了，是世界上最可爱的罗琳走到了我面前"
    words = jieba.lcut(text)
    print(f"分词结果: {words}")

    # 创建词嵌入层 参一：词表大小 参二：词向量的维度
    embed = nn.Embedding(len(words), 4)
    #enumerate()：返回列表中的每个值和其对应的索引
    for i, word in enumerate(words):
        print(i, word)
        word_vec = embed(torch.tensor(i))
        print(word_vec)
        print()

dm()