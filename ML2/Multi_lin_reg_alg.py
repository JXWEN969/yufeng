# 导入库和模块
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_validate
import torch
import torch.utils.data as Data
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import preprocessing
import matplotlib.pyplot as plt 
import seaborn as sns 

# 读取数据文件
data_df = pd.read_csv('D:\桌面\基于回归分析的大学综合得分预测(1)\基于回归分析的大学综合得分预测\cwurData.csv')  # 读入 csv 文件为 pandas 的 DataFrame
#print(data_df.head(3).T)  # 观察前几列并转置方便观察
data_df = data_df.dropna()  # 舍去包含 NaN 的 row

# 对 region 列进行离散化处理
region_dummies = pd.get_dummies(data_df['region'])

# 将生成的新的列添加到原来的数据中，并且删除原来的 region 列
data_df = pd.concat([data_df, region_dummies], axis=1)
data_df = data_df.drop('region', axis=1)


# 选择特征列
feature_cols = ['quality_of_faculty', 'publications', 'citations', 'alumni_employment',
                'influence', 'quality_of_education', 'broad_impact', 'patents']
X = data_df[feature_cols]
Y = data_df['score']
all_y = Y.values
all_x = X.values

# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(all_x, all_y, test_size=0.2)

# 最小二乘法
feature = torch.tensor(x_train,dtype=torch.float)
lable = torch.tensor(y_train,dtype=torch.float)#numpy转tensor
bais = torch.ones(feature.shape[0])
bais = bais.view([1600,1])
feature = torch.cat((feature,bais),1)#为特征增添偏移量
x_t = torch.tensor(x_test,dtype=torch.float)
baiss = torch.ones(x_t.shape[0])
baiss = baiss.view([len(x_t),1])
x_t = torch.cat((x_t,baiss),1)
w_pre = torch.mm(torch.mm(torch.inverse(torch.mm(torch.t(feature),feature)),torch.t(feature)),lable.view(1600,1))#核心，用最小二乘计算w
print("最小二乘法的参数为：", list(w_pre))
loss = torch.tensor(y_test).view([400,1]) - torch.mm(x_t,w_pre)
loss = torch.mm(torch.t(loss),loss)
RMSE = (loss.item()/len(lable))**0.5
print("最小二乘法的 RMSE 为：", RMSE)
var = (torch.tensor(y_test)-torch.mean(torch.tensor(y_test))).view([-1,1])
var = torch.mm(torch.t(var),var)
R = 1-loss.item()/var.item()
print("最小二乘法的 R^2 为：", R)

# 梯度下降法
min_max_scaler = preprocessing.MinMaxScaler() # 定义数据归一化的变量
all_x = min_max_scaler.fit_transform(all_x)#数据归一化
x_train, x_test, y_train, y_test = train_test_split(all_x, all_y, test_size=0.2)#将数据分为训练集与测试集
feature = torch.tensor(x_train,dtype=torch.float)
lable = torch.tensor(y_train,dtype=torch.float)#numpy转tensor
bais = torch.ones(feature.shape[0])
bais = bais.view([1600,1])
feature = torch.cat((feature,bais),1)#为特征增添偏移量
x_t = torch.tensor(x_test,dtype=torch.float)
baiss = torch.ones(x_t.shape[0])
baiss = baiss.view([len(x_t),1])
x_t = torch.cat((x_t,baiss),1)
w_pre = torch.rand(len(feature_cols)+1,1)
num_epoch = 500
lr = 0.3
for i in range(1,num_epoch+1):
        y_pre = torch.mm(feature,w_pre)
        loss = y_pre - lable.view([1600,1])
        loss = loss/len(feature)
        loss = torch.mm(torch.t(feature),loss)
        w_pre = w_pre - lr*loss#核心部分，进行梯度下降
y_pre = torch.mm(x_t,w_pre)
loss = torch.tensor(y_test).view([400,1]) - y_pre
loss = torch.mm(torch.t(loss),loss)
rms = (loss.item()/len(y_pre))**0.5
print("梯度下降法的参数为：", list(w_pre))
print("梯度下降法的 RMSE 为：", rms)#计算RMSE
var = (torch.tensor(y_test)-torch.mean(torch.tensor(y_test))).view([-1,1])
var = torch.mm(torch.t(var),var)
R = 1-loss.item()/var.item()
print("梯度下降法的 R^2 为：", R)#计算R^2

# sklearn 库的函数
reg = linear_model.LinearRegression()
reg.fit(x_train,y_train)#训练
y_pre=reg.predict(x_test)#预测
rmse = mean_squared_error(y_test,y_pre,squared = False)
print("sklearn 库的函数的参数为：", list(reg.coef_) + [reg.intercept_])
print("sklearn 库的函数的 RMSE 为：", rmse)
print("sklearn 库的函数的 R^2 为：", r2_score(y_test,y_pre))#评估

# 数据可视化
# 绘制前十五名学校的综合得分的条形图
mean_df = data_df.groupby('institution').mean()  # 按学校聚合并对聚合的列取平均
top_15_df = mean_df.sort_values(by='score', ascending=False).head(15)  # 取前十五学校
sns.set()
x = top_15_df['score'].values  # 综合得分列表
y = top_15_df.index.values  # 学校名称列表
sns.barplot(x=x, y=y, orient='h', palette="Reds_d", hue=y, legend=False)  
plt.xlim(75, 101)  
plt.title('TOP 15') 
plt.xlabel('SCORE') 
plt.ylabel('SCHOOL') 
plt.show()

# 绘制各个特征与综合得分的散点图
sns.pairplot(data_df, x_vars=feature_cols, y_vars='score', kind='reg', height=3)
plt.suptitle('Relationship between Features and Score') 
plt.show()

# 绘制各个特征之间的相关系数热力图
corr = data_df[feature_cols].corr() # 计算各个特征之间的相关系数
sns.heatmap(corr, annot=True, cmap='Blues') 
plt.title('Correlation Matrix of Features')
plt.show()