import csv
from gensim.models import word2vec

address1 = "/media/kerrwy/software/python/works/Chinese text classify/data/sample_training.csv"
address2 = "/media/kerrwy/software/python/works/Chinese text classify/data/sample_testing.csv"

with open(address1, 'r') as file:
    train = csv.reader(file)
    tr = [tr for tr in train]    # 训练数据列表
print(tr)

with open(address2, 'r') as file:
    test = csv.reader(file)
    te = [te for te in test]    # 测试数据列表


x_train = []
for i in range(len(tr)):
    x_train.append(tr[i][1])
print(x_train)


# 词向量技术
num_feature = 300  # 词向量的维度
min_word_count = 10  # 保证被考虑的词汇的频度
num_workers = 2  # 设定并行化训练使用CPU计算核心的数量，多核可用
context = 4  # 定义训练词向量的上下文窗口大小
downsampling = 1e-3

model = word2vec.Word2Vec(x_train, workers=num_workers,\
                          size=num_feature, min_count=min_word_count,\
                          window=context, sample=downsampling)
model.init_sims(replace=True)  # 加快模型训练速度，当前训好的词向量为最终版
model.build_vocab_from_freq()