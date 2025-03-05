# 关于

本仓库是对以下实验内容的汇总：


## 机器学习

- [实验1](./MachineLearning/ML1/README.md)

- [实验2](./MachineLearning/ML2/README.md)

- [实验3 (TODO)](./MachineLearning/ML3/README.md) 

- [实验4](./MachineLearning/ML4/README.md)

- [实验5](./MachineLearning/ML5/README.md)

- [实验6](./MachineLearning/ML6/README.md)


## 深度学习

- [案例1](./DeepLearning/DL1/README.md)

- [案例2](./DeepLearning/DL2/README.md)

- [案例3](./DeepLearning/DL3/README.md)

- [案例4](./DeepLearning/DL4/README.md)

- [案例5](./DeepLearning/DL5/README.md)

- [案例6](./DeepLearning/DL6/README.md)

- [案例7](./DeepLearning/DL7/README.md)



# 一、 选题背景：
决策树是一种常用的机器学习算法，它可以根据一系列的特征和规则，将数据划分为不同的类别。决策树的优点是易于理解和解释，可以处理离散和连续的特征，可以处理缺失值和噪声数据，可以进行特征选择和剪枝等操作。决策树的缺点是容易过拟合，对于小的变化敏感，可能产生不稳定的结果，可能不是最优的解决方案。

ID3算法是一种生成决策树的经典算法，它基于信息增益来选择最佳的划分特征，递归地构建决策树。ID3算法的优点是简单直观，可以处理多值的特征，可以生成简洁的决策树。ID3算法的缺点是不能处理连续的特征，不能处理缺失值，对于不平衡的数据分布敏感，容易产生过拟合。

排位赛的数据集是一个适合用决策树进行分类的数据集，因为它包含了多个离散的特征，以及一个明确的分类标签。


# 二、 实验目的及要求
本实验的目的是使用Python语言实现决策树的ID3算法，并用它来对排位赛的数据集进行分类，预测比赛的胜负结果。

本实验的要求是：
编写一个决策树的类，定义初始化、训练、预测、分裂、遍历、计算熵等方法。
读取数据集，对数据进行预处理，如删除无关的特征，离散化连续的特征，划分训练集和测试集等。

使用决策树的类，对训练集进行训练，得到一个决策树的模型。

使用决策树的类，对测试集进行预测，得到一个预测的结果。

计算预测结果的准确率，与sklearn库中的决策树模型进行对比，分析模型的效果和效率。

撰写实验报告，包括实验的背景、目的、原理、方法、结果、分析、结论等内容。

# 三、实验原理
本实验的原理是决策树的ID3算法，它的主要步骤如下：
从根节点计算熵，算特征下的信息增益，选择信息增益最大的特征作为当前节点的划分特征，根据该特征的不同取值，将数据集划分为若干子集。选择信息增益最大的特征，如果子集中只有一个类别，或者达到了最大深度，或者小于最小划分数，就将子集作为一个叶子节点，标记为该类别；否则，递归地对子集进行上述步骤，生成子节点。
返回根节点，得到一个决策树的模型。
对于一个新的数据，从根节点开始，根据其特征值，沿着决策树的路径，直到达到一个叶子节点，返回该节点的类别，作为预测的结果。

# 四、代码展示：
```python
from collections import Counter
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier # 导入sklearn库中的决策树模型，作为对比
import time # 导入time库，用于计算运行时间
import matplotlib.pyplot as plt

RANDOM_SEED =2100

# 读取CSV文件
csv_data = "D:\桌面\high_diamond_ranked_10min.csv"
data_df = pd.read_csv(csv_data, sep=',')
data_df = data_df.drop(columns='gameId')  # 删除不需要的列

# 打印第一行数据和数据描述
print(data_df.iloc[0])
data_df.describe()

# 删除一些特征列
drop_features = ['blueGoldDiff', 'redGoldDiff', 
                 'blueExperienceDiff', 'redExperienceDiff', 
                 'blueCSPerMin', 'redCSPerMin', 
                 'blueGoldPerMin', 'redGoldPerMin']
df = data_df.drop(columns=drop_features)

# 创建新的特征列，计算蓝队和红队的差值
info_names = [c[3:] for c in df.columns if c.startswith('red')]
for info in info_names:
    df['br' + info] = df['blue' + info] - df['red' + info]
df = df.drop(columns=['blueFirstBlood', 'redFirstBlood'])  # 删除不需要的列

# 对数据进行离散化处理
discrete_df = df.copy()
for c in df.columns[1:]:
    discrete_df[c] = pd.qcut(df[c], q=5, labels=False, duplicates='drop')

# 准备训练数据和标签
all_y = discrete_df['blueWins'].values
feature_names = discrete_df.columns[1:]
all_x = discrete_df[feature_names].values

# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(all_x, all_y, test_size=0.2, random_state=RANDOM_SEED)
all_y.shape, all_x.shape, x_train.shape, x_test.shape, y_train.shape, y_test.shape

# 自定义决策树类
class DecisionTree(object):
    def __init__(self, classes, features, max_depth=10, min_samples_split=10, impurity_t='entropy'):
        self.classes = classes  # 类别
        self.features = features  # 特征
        self.max_depth = max_depth  # 最大深度
        self.min_samples_split = min_samples_split  # 最小样本分割数
        self.impurity_t = impurity_t  # 不纯度类型
        self.root = None  # 根节点

    def fit(self, feature, label):
        assert len(self.features) == len(feature[0])  # 确保特征数量一致
        self.root = self.expand_node(feature, label, depth=1)  # 构建决策树

    def predict(self, feature):
        assert len(feature.shape) == 1 or len(feature.shape) == 2  # 确保特征维度正确
        if len(feature.shape) == 1:
            return self.traverse_node(self.root, feature)  # 遍历节点进行预测
        return np.array([self.traverse_node(self.root, f) for f in feature])  # 批量预测

    def expand_node(self, feature, label, depth):
        if len(set(label)) == 1:  # 如果所有标签相同，返回该标签
            return {'class': label[0]}
        if depth == self.max_depth:  # 如果达到最大深度，返回最常见的标签
            return {'class': Counter(label).most_common(1)[0][0]}
        if len(feature) < self.min_samples_split:  # 如果样本数小于最小分割数，返回最常见的标签
            return {'class': Counter(label).most_common(1)[0][0]}
        
        entropy = self.calc_entropy(label)  # 计算熵
        info_gains = [entropy - self.calc_cond_entropy(feature, label, i) for i in range(len(self.features))]  # 计算信息增益
        best_feature_index = np.argmax(info_gains)  # 选择信息增益最大的特征
        best_feature_name = self.features[best_feature_index]
        best_feature_values = set(feature[:, best_feature_index])
        
        node = {'feature_index': best_feature_index, 'feature_name': best_feature_name, 'children': {}}
        global sub_label
        for value in best_feature_values:
            sub_feature = feature[feature[:, best_feature_index] == value]
            sub_label = label[feature[:, best_feature_index] == value]
            sub_node = self.expand_node(sub_feature, sub_label, depth + 1)  # 递归构建子树
            node['children'][value] = sub_node
        return node

    def traverse_node(self, node, feature):
        label = sub_label # 获取对应的标签子集
        if 'class' in node:  # 如果节点是叶子节点，返回类别
            return node['class']
        feature_index = node['feature_index']
        feature_name = node['feature_name']
        feature_value = feature[feature_index]
        if feature_value not in node['children']:  # 如果特征值不在子节点中，返回最常见的标签
            return Counter(label).most_common(1)[0][0]
        sub_node = node['children'][feature_value]
        return self.traverse_node(sub_node, feature)  # 递归遍历子树

    def calc_cond_entropy(self, feature, label, index):
        feature_values, feature_counts = np.unique(feature[:, index], return_counts=True)
        cond_entropy = 0.0
        for value, count in zip(feature_values, feature_counts):
            sub_label = label[feature[:, index] == value]
            sub_entropy = self.calc_entropy(sub_label)  # 计算子集的熵
            prob = count / len(label)
            cond_entropy += prob * sub_entropy  # 计算条件熵
        return cond_entropy

    def calc_entropy(self, label):
        counter = Counter(label)
        probs = [count / len(label) for count in counter.values()]
        entropy = -sum([p * np.log2(p) for p in probs])  # 计算熵
        return entropy

# 创建自己的决策树模型
my_model = DecisionTree(classes=[0, 1], features=feature_names)
# 记录开始时间
start_time = time.time()
# 训练模型
my_model.fit(x_train, y_train)
# 预测测试集
y_pred = my_model.predict(x_test)
# 计算准确率
my_acc = accuracy_score(y_test, y_pred)
# 记录结束时间
end_time = time.time()
# 打印结果
print('My model accuracy:', my_acc)
print('My model time:', end_time - start_time)

# 创建sklearn的决策树模型
sk_model = DecisionTreeClassifier()
# 记录开始时间
start_time = time.time()
# 训练模型
sk_model.fit(x_train, y_train)
# 预测测试集
y_pred = sk_model.predict(x_test)
# 计算准确率
sk_acc = accuracy_score(y_test, y_pred)
# 记录结束时间
end_time = time.time()
# 打印结果
print('Sklearn model accuracy:', sk_acc)
print('Sklearn model time:', end_time - start_time)

# 绘制游戏结果的分布图
plt.hist(data_df['blueWins'], bins=2, color='blue', label='Blue Wins')
plt.xlabel('Win or Lose')
plt.ylabel('Count')
plt.title('Distribution of Game Results')
plt.legend()
plt.show()
```
