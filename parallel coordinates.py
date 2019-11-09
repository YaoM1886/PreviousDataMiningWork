import pandas as pd
import numpy as np
import time
import graphviz
import random
from sklearn  import tree
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from matplotlib import mlab
from matplotlib.colors import ListedColormap
from sklearn.model_selection import cross_val_score
from pandas.plotting import parallel_coordinates
import scipy.stats as sts



#中心位置
# np.mean(data)#均值
# np.median(data)#中位数
# sts.mode(data)#众数

#len(data)#数据个数


#发散程度
# ptp(data)
# np.var(data)
# np.std(data)
# std(data)／mean(data)#变异系数
#sts.quantile(data, p=0.75) - sts.quantile(data, p=0.25)#四分位差
#
#偏峰与峰度
#sts.skewness(data)#偏度大于0是右偏
#sts.kurtosis(data)#峰度，正态分布的峰度是3，大于3说明观察更集中，有比正态更短的尾部





# #Z分数衡量偏差程度，测量值距离均值相差的标准差的数量，一般绝对值不要大于3
# (data[*] - mean(data)) / std(data)
#
# #计算协方差看是否相关,返回结果是矩阵，第i行第j列数据表示i和j组数的协方差
# cov(data, bias=1)
# #计算相关系数
# corrcoef(data)

#箱型图
# def drawbox(heights):
#     plt.boxplot([heights], labels=['Heights'])
#     plt.title("heights of male students")
#     plt.show()


#二项分布中的size表示采样的次数，np.random实际上采用的是统计方法进行采样，把造好的数据存储在列表里，这样之后就不每次都变化随机了
mu = 0
sigma = 1
size = 300
lamb = 10
n = 10
p = 0.5
x1 = np.random.normal(mu, sigma, size=size)
x2 = np.random.poisson(lamb, size=size)
x3 = np.random.exponential(lamb, size=size)
x4 = np.random.binomial(n, p, size=size)
y = np.random.binomial(1, p, size=size)



source = {'normal_data': x1,
          'normal_data2': x1,
        'poisson_data':x2,
        'exponential_data2':x3,
        'exponential_data3':x3,
        'normal_data3':x1,
        'normal_data4':x1,
        'normal_data5':x1,
        'normal_data6':x1,
        'normal_data7':x1,
        'normal_data8':x1,
        'normal_data9':x1,
        'exponential_data4':x3,
        'exponential_data5': np.random.exponential(10, size=300),
        'exponential_data6': np.random.exponential(10, size=300),
        'normal_data10': np.random.normal(mu, sigma, size=300),
        'normal_data11': np.random.normal(mu, sigma, size=300),
        'normal_data12': np.random.normal(mu, sigma, size=300),
        'normal_data13': np.random.normal(mu, sigma, size=300),
        'normal_data14': np.random.normal(mu, sigma, size=300),
        "binomial_data":x4,



        'normal_data111': x1,
          'normal_data222': x1,
        'poisson_data11':x2,
        'exponential_data22':x3,
        'exponential_data33':x3,
        'normal_data33':x1,
        'normal_data44':x1,
        'normal_data55':x1,
        'normal_data66':x1,
        'normal_data77':x1,
        'normal_data88':x1,
        'normal_data99':x1,
        'exponential_data44':x3,
        'exponential_data55': np.random.exponential(10, size=300),
        'exponential_data66': np.random.exponential(10, size=300),
        'normal_data101': np.random.normal(mu, sigma, size=300),
        'normal_data110': np.random.normal(mu, sigma, size=300),
        'normal_data121': np.random.normal(mu, sigma, size=300),
        'normal_data131': np.random.normal(mu, sigma, size=300),
        'normal_data141': np.random.normal(mu, sigma, size=300),
        "binomial_data00":x4,
        'class': np.random.binomial(1, 0.5, size=300)}
result = pd.DataFrame(source)
X = result[["normal_data", "normal_data2", 'poisson_data', 'exponential_data2', 'exponential_data3', 'exponential_data4',
            'exponential_data5', 'exponential_data6', 'normal_data3', 'normal_data4', 'normal_data5', 'normal_data6',
            'normal_data7', 'normal_data8', 'normal_data9', 'normal_data10', 'normal_data11', 'normal_data12', 'normal_data13',
            'normal_data14', 'binomial_data','normal_data111',  'normal_data222', 'poisson_data11', 'exponential_data22', 'exponential_data33',
            'normal_data33','normal_data44', 'normal_data55','normal_data66','normal_data77','normal_data88','normal_data99',
            'exponential_data44','exponential_data55',  'exponential_data66', 'normal_data101',
        'normal_data110',
        'normal_data121',
        'normal_data131',
        'normal_data141',
        "binomial_data00"]]
y = result['class']


#画出原始的二维特征散点图
plt.scatter(x3, x4, marker='o', c=y)
plt.title("unpredicted pic")
plt.show()

#平行坐标的绘制
# feature_names = ["normal_data", "normal_data2", "exponential_data2", "poisson_data"]
# data_dict = {}
# column = 0
# for feature_name in feature_names:
#     data_dict[feature_name] = X.iloc[:, column]
#     column += 1
# pd_data = pd.DataFrame(data_dict)
# target_names = ["good", "bad"]
# target_labels = []
# for class_num in y:
#     target_labels.append(target_names[class_num])
# data_dict["target_labels"] = target_labels
plt.figure()
parallel_coordinates(result, result["class"])
plt.show()



#模型拟合
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=33)
vec = DictVectorizer(sparse=False)
X_train = vec.fit_transform(X_train.to_dict(orient="record"))
X_test = vec.fit_transform(X_test.to_dict(orient="record"))
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)


start = time.time()


knn = KNeighborsClassifier(weights='distance', algorithm='brute')
knn.fit(X_train, y_train)
y_predict = knn.predict(X_test)
print("The accuracy of knn is:", knn.score(X_test, y_test))
print(classification_report(y_test, y_predict, target_names=['good', 'bad']))

print(time.time()-start)


#找出最优的K值
accuracy = []
for i in np.arange(20)+1:
    knn = KNeighborsClassifier(n_neighbors=i)
    score1 = cross_val_score(knn, X, y, cv=10)
    scores = score1.mean()
    accuracy.append(scores)
xs = np.arange(len(accuracy)) + 1
plt.plot(xs, accuracy, color='c')
plt.xlabel('K')
plt.ylabel('accuracy ratio')
plt.title('optimal K')
plt.show()

# # #画出彩色分类决策边界
#
# cmap_light = ListedColormap(['#AAAAFF', '#AAFFAA', '#FFAAAA'])
# cmap_bold = ListedColormap(['#FF0000', '#003300', '#0000FF'])
# h = 0.02
# x_min, x_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
# y_min, y_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1
# xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
#                      np.arange(y_min, y_max, h)) #生成网格型二维数据对
# print(np.c_[xx.ravel(), yy.ravel()])
# Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
#
# plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
# plt.scatter(X[:, 0], X[:, 1], marker='o', c=y, cmap=cmap_bold)
# plt.show()



