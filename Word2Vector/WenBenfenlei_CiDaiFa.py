#import pandas as pd
import csv
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

# only use sample_training to train
# training = pd.read_csv('/media/kerrwy/software/python/works/Chinese text classify/data/sample_training.csv')
#
# print(len(training), type(training), training)
# value = training.values
# print(value)
# print(type(value))
# print(value.shape)  # 19rows, DataFrame will reduce one hang.



address1 = "/media/kerrwy/software/python/works/Chinese text classify/data/sample_training.csv"
address2 = "/media/kerrwy/software/python/works/Chinese text classify/data/sample_testing.csv"


# 训练样本
with open(address1, 'r') as file:
    training = csv.reader(file)
    rows = [row for row in training]  # 将样本转化为多个列表，每个列表中包含标签和文本

print(training)
print(rows)
print(type(rows))
print(len(rows))
print(rows[0][1])


x_training = []
y_training = []
for i in range(20):
    x_training.append(rows[i][1])  # 只提取列表中的文字部分，
    y_training.append(rows[i][0])  # 只提取标签

# print(x_training)
# print(y_training)
# print(len(x_training))


# 提取样本中的词汇特征，将其向量化
count_vec = CountVectorizer()
x_count_train = count_vec.fit_transform(x_training)  # there needs DataFrame
# x_count_train = count_vec.fit_transform(['a dog eat',  'baby sleep', 'I want to go home'])

# print(x_count_train)
print(np.shape(x_count_train), (x_count_train.toarray()))
print(count_vec.get_feature_names())



# 测试样本
with open(address2, 'r') as file:
    testing = csv.reader(file)
    test = [test for test in testing]
print(test)


x_testing = []
for i in range(len(test)):
    x_testing.append(test[i][1])


x_count_test = count_vec.fit_transform(x_testing)

print(len(test))
print(np.shape(x_count_test.toarray()), (x_count_test.toarray()))
print(count_vec.get_feature_names())
print(np.max(x_count_test.toarray()))

# 构造模型

# 朴素贝叶斯分类器
from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB()
mnb.fit(x_count_train, y_training)
y_predict = mnb.predict(x_count_test)
print(y_predict)

# # SVM
# lsvc = LinearSVC()
# lsvc.fit(x_count_train, y_training)
# y_predict = lsvc.predict(x_count_test)
# print(y_predict)
#
# # 评测
# print('准确率：',lsvc.score(x_count_test, y_testing))  # 缺少测试集标签
# print("查准率和查全率：", classification_report(y_testing, y_predict))

