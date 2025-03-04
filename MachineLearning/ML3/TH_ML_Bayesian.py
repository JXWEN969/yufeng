# 导入工具
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, ComplementNB # 导入三种朴素贝叶斯分类器
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix # 导入评估指标
from sklearn.model_selection import train_test_split # 导入数据划分方法

# 加载数据
label_path = r"D:\桌面\贝叶斯垃圾邮件识别\hw3\trec06c-utf8\label\index"
data_path = r"D:\桌面\贝叶斯垃圾邮件识别\hw3\trec06c-utf8\data_cut"
#配置中文字体
plt.rcParams["font.family"] = "SimSun"
# 提取标签
labels = []
with open(label_path, 'r', encoding='gbk') as f:
    for line in f.readlines():
        label, path = line.strip().split()
        labels.append(label)

# 统计标签
print('邮件总数:', len(labels))
print('垃圾邮件数:', labels.count('spam'))
print('正常邮件数:', labels.count('ham'))

# 获取文件路径
def get_file_path(label_path):
    file_paths = []
    with open(label_path, 'r', encoding='gbk') as f:
        for line in f.readlines():
            label, path = line.strip().split()
            file_paths.append(path)
    return file_paths

file_paths = get_file_path(label_path)

# 提取邮件
mailHeader_list = []
mailContent_list = []
for file_path in file_paths:
    with open(data_path + file_path[7:], 'r', encoding='gbk', errors='ignore') as f:
        mail = f.read()
        mailHeader, mailContent = mail.split('\n\n', 1) # 分割邮件头和邮件内容
        mailHeader_list.append(mailHeader)
        mailContent_list.append(mailContent)

# 打印一封邮件
print('标签:', labels[0])
print('邮件头:\n', mailHeader_list[0])
print('邮件内容:\n', mailContent_list[0])

# 文本特征提取
max_features = 7000
vectorizer = TfidfVectorizer() # 使用TF-IDF方法提取文本特征
X = vectorizer.fit_transform(mailContent_list) # 将邮件内容转化为词向量
y = np.array(labels) # 将标签转化为数组
print('特征维度:', X.shape[1])
print('特征示例:\n', X[0])

# 划分训练集和测试集
np.random.seed(2020) # 设置随机种子
train_index = np.random.choice(range(len(y)), int(len(y) * 0.8), replace=False) # 随机选择80%的数据作为训练集
test_index = list(set(range(len(y))) - set(train_index)) # 剩余的20%的数据作为测试集
X_train = X[train_index] # 训练集特征
y_train = y[train_index] # 训练集标签
X_test = X[test_index] # 测试集特征
y_test = y[test_index] # 测试集标签

# 使用朴素贝叶斯算法进行分类
clf = MultinomialNB() # 创建朴素贝叶斯分类器
clf.fit(X_train, y_train) # 训练模型
y_pred = clf.predict(X_test) # 预测测试集

# 评估模型效果
acc = accuracy_score(y_test, y_pred) # 准确率
pre = precision_score(y_test, y_pred, pos_label='spam') # 精准率
rec = recall_score(y_test, y_pred, pos_label='spam') # 召回率
f1 = f1_score(y_test, y_pred, pos_label='spam') # F1值
print('准确率:', acc)
print('精准率:', pre)
print('召回率:', rec)
print('F1值:', f1)

# 使用不同的朴素贝叶斯分类器进行比较
MNB = MultinomialNB() # 多项式朴素贝叶斯分类器
BNB = BernoulliNB() # 伯努利朴素贝叶斯分类器
CNB = ComplementNB() # 补码朴素贝叶斯分类器

accuracy_list, precision_list, recall_list = [],[],[] # 存储不同分类器的评估指标
for model_name,model in zip(['多项式朴素贝叶斯分类器','伯努利朴素贝叶斯分类器','补码朴素贝叶斯分类器'],[MNB,BNB,CNB]):
    model.fit(X_train, y_train) # 训练过程 
    y_pred= model.predict(X_test) # 在测试集上预测
    model_accuracy_score = accuracy_score(y_test,y_pred).round(4) # 在测试集上计算准确率
    model_precision_score = precision_score(y_test,y_pred, pos_label='spam').round(4) # 在测试集上计算精准率
    model_recall_score = recall_score(y_test,y_pred, pos_label='spam').round(4) # 在测试集上计算召回率
    accuracy_list.append(model_accuracy_score)
    precision_list.append(model_precision_score)
    recall_list.append(model_recall_score)
    print(model_name+'准确率为:'+str(model_accuracy_score))
    print(model_name+'精准率为:'+str(model_precision_score))
    print(model_name+'召回率为:'+str(model_recall_score))

# 使用不同的特征维度进行比较
def word_to_vector(mail_list,max_features:int):
    '''
    对正文部分进行特征提取，指定方法：
        count不考虑词频，tfidf考虑词频，使用tfidf计算
    '''
    wov_model = TfidfVectorizer(token_pattern=r"(?u)\b\w\w+\b",min_df=5, 
                                max_df=0.6, max_features=max_features, 
                                ngram_range=(1, 1))
    mail_tfidf = wov_model.fit_transform(mail_list)

    return  mail_tfidf

max_features_list = [50, 100, 500, 1000, 2000, 3000, 5000, 7000] # 不同的特征维度
CNB_accuracy_list, CNB_precision_list, CNB_recall_list = [],[],[] # 存储不同特征维度下的评估指标
for max_features in max_features_list:
    X_new=word_to_vector(mail_list=mailContent_list,max_features=max_features) # 提取新的特征
    X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2, random_state=2022) # 划分数据集
    CNB.fit(X_train, y_train)  #训练过程 
    y_pred= CNB.predict(X_test) #在测试集上预测
    CNB_accuracy_score = accuracy_score(y_test,y_pred).round(4)  #在测试集上计算准确率
    CNB_precision_score = precision_score(y_test,y_pred, pos_label='spam').round(4) # 在测试集上计算精准率
    CNB_recall_score = recall_score(y_test,y_pred, pos_label='spam').round(4) # 在测试集上计算召回率
    print('max_feature: '+str(max_features))
    CNB_accuracy_list.append(CNB_accuracy_score)
    CNB_precision_list.append(CNB_precision_score)
    CNB_recall_list.append(CNB_recall_score)
    print('tfidf+CNB模型的准确率为:'+str(CNB_accuracy_score))
    print('tfidf+CNB模型的精度为:'+str(CNB_precision_score))
    print('tfidf+CNB模型的召回率为:'+str(CNB_recall_score))

# 将不同特征维度下的评估指标展示在表格中
data={"max_feature":max_features_list,
      "准确率":CNB_accuracy_list,
      "精度":CNB_precision_list,
      "召回率":CNB_recall_list}

result_df = pd.DataFrame(data)
result_df

# 绘制柱状图，比较不同的朴素贝叶斯分类器的评估指标
plt.figure(figsize=(12, 8)) # 设置画布大小
x = np.arange(3) # 设置柱状图的位置
width = 0.2 # 设置柱状图的宽度
plt.bar(x, accuracy_list, width, label='准确率') # 绘制准确率柱状图
plt.bar(x + width, precision_list, width, label='精度') # 绘制精度柱状图
plt.bar(x + 2 * width, recall_list, width, label='召回率') # 绘制召回率柱状图
plt.xticks(x + width, ['多项式朴素贝叶斯分类器','伯努利朴素贝叶斯分类器','补码朴素贝叶斯分类器']) # 设置x轴刻度
plt.ylim(0.9, 1) # 设置y轴范围
plt.xlabel('分类器') # 设置x轴标签
plt.ylabel('评估指标') # 设置y轴标签
plt.title('不同的朴素贝叶斯分类器的评估指标比较') # 设置标题
plt.legend() # 显示图例
plt.show() # 显示图形

# 绘制折线图，分析最大文本特征数量对模型性能的影响
plt.figure(figsize=(12, 8)) # 设置画布大小
plt.plot(max_features_list, CNB_accuracy_list, label='准确率') # 绘制准确率折线图
plt.plot(max_features_list, CNB_precision_list, label='精度') # 绘制精度折线图
plt.plot(max_features_list, CNB_recall_list, label='召回率') # 绘制召回率折线图
plt.xscale('log') # 设置x轴为对数刻度
plt.xlabel('最大文本特征数量') # 设置x轴标签
plt.ylabel('评估指标') # 设置y轴标签
plt.title('最大文本特征数量对模型性能的影响') # 设置标题
plt.legend() # 显示图例
plt.show() # 显示图形


