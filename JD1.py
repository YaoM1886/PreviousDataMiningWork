import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
import missingno as msno
import seaborn as sn
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeRegressor





# #从基本信息表格中把不同第一类目的产品挑选出来，存到first_cate==1／6／7／的表格里
# info_dataframe = pd.read_csv("/Users/mac/文档/京东论文/forecast_data_0523/sku_info.csv")
# def get_firstcate():
#    for i in [1, 6, 7]:
#         info1 = info_dataframe[info_dataframe['item_first_cate_cd'] == i]
#         info1.to_csv("/Users/mac/文档/京东论文/forecast_data_0523/first_cate==%d.csv"%i)
# get_firstcate()
#
#
# #将三个一级类目的表和销量表进行交连接,需要折扣列
# info3 = pd.read_csv("/Users/mac/文档/京东论文/forecast_data_0523/sku_sales.csv")
# def inner_cate_sales():
#     for i in [1, 6, 7]:
#         df1 = pd.read_csv("/Users/mac/文档/京东论文/forecast_data_0523/first_cate==%d.csv"%i)
#         df = pd.merge(df1, info3, how='left', on='item_sku_id')
#         df.drop(['vendibility'], axis=1, inplace=True)
#         for j in np.arange(0, 6):
#             form = df[df['dc_id'] == j]
#             form.drop(['dc_id', 'item_first_cate_cd'], axis=1, inplace=True)
#             form.set_index('item_sku_id', inplace=True)
#             form.to_csv("/Users/mac/文档/京东论文/forecast_data_0523/cate_dc_sales/cate%d_dc%d_sales.csv"%(i, j))
# inner_cate_sales()

#销量列、促销类型列，但是差折扣列
# for i in np.arange(0, 6):
#     view = pd.read_csv("/Users/mac/文档/京东论文/training/sales%d_train.csv"%i)
#     for j in [1, 6, 7]:
#         fisrt = pd.read_csv("/Users/mac/文档/京东论文/forecast_data_0523/first_cate==%d.csv"%j)
#         sku1 = fisrt["item_sku_id"]
#         ls = list(sku1)
#         sub_df = view[view["item_sku_id"].isin(ls)]
#         sub_df.to_csv("/Users/mac/文档/京东论文/forecast_data_0523/sales_prom/cate%d_dc%d_sales_prom.csv"%(j, i))

#把两个文件夹的15个表连接起来，加上折扣列
# for i in [1, 6, 7]:
#     for j in np.arange(0, 6):
#         converge1 = pd.read_csv("/Users/mac/文档/京东论文/forecast_data_0523/sales_prom/cate%d_dc%d_sales_prom.csv"%(i, j))
#         converge1.drop(["Unnamed: 0"], inplace=True, axis=1)
#         converge2 = pd.read_csv("/Users/mac/文档/京东论文/forecast_data_0523/cate_dc_sales/cate%d_dc%d_sales.csv"%(i, j))
#         #哭了，以后日期再不格式化就是傻子！！！！！！！！！！！
#         converge1["date"] = pd.to_datetime(converge1["date"], format="%Y-%m-%d")
#         converge2["date"] = pd.to_datetime(converge2["date"], format="%Y-%m-%d")
#         converge = pd.merge(converge1, converge2, how='outer')
#         converge.to_csv("/Users/mac/文档/京东论文/forecast_data_0523/sales_promo_quan/cate%d_dc%d_sales_prom_quan.csv"%(i, j))

#对15张表的销量进行描述性统计，观察销量数目的分布，为之后的销量分高低做铺垫
# def descriptive_quantity():
#     for i in [1, 6, 7]:
#         for j in np.arange(0, 6):
#             exper = pd.read_csv('/Users/mac/文档/京东论文/forecast_data_0523/sales_prom_quan/cate%d_dc%d_sales_prom_quan.csv'%(i, j))
#             print("第%d类目的产品在%d分销地的描述性统计量是:"%(i, j), exper['quantity'].describe())
#             print("\n\n\n")
# descriptive_quantity()

# #画出销量分布的直方图观察一下
# def plot_quantity():
#     plt.subplot(3, 5, 1)
#     exper = pd.read_csv('/Users/mac/文档/京东论文/forecast_data_0523/sales_prom_quan/cate1_dc0_sales_prom_quan.csv')
#     print(exper['quantity'].describe())
#     plt.hist(exper['quantity'], bins=20, color='steelblue', edgecolor='k')
#
#
#
#     plt.subplot(3, 5, 2)
#     exper = pd.read_csv('/Users/mac/文档/京东论文/forecast_data_0523/sales_prom_quan/cate1_dc1_sales_prom_quan.csv')
#     plt.hist(exper['quantity'], bins=3, color='steelblue', edgecolor='k')
#
#
#     plt.subplot(3, 5, 3)
#     exper = pd.read_csv('/Users/mac/文档/京东论文/forecast_data_0523/sales_prom_quan/cate1_dc2_sales_prom_quan.csv')
#     plt.hist(exper['quantity'], bins=3, color='steelblue', edgecolor='k')
# plot_quantity()

#读取各个数据表，做预处理
#去除未知列
#将日期转化成标准格式
#将折扣NA的值都填充成0
def readForm(i, j):
    inputForm = pd.read_csv('/Users/mac/文档/京东论文/forecast_data_0523/sales_promo_quan/cate%d_dc%d_sales_prom_quan.csv'%(i, j))
    inputForm.drop(["Unnamed: 0.1", 'Unnamed: 0'], inplace=True, axis=1)
    inputForm['date'] = pd.to_datetime(inputForm['date'], format='%Y-%m-%d')
    # inputForm = inputForm.fillna({'discount': 0})
    inputForm['year'] = pd.DatetimeIndex(inputForm['date']).year
    inputForm['month'] = pd.DatetimeIndex(inputForm['date']).month
    inputForm['day'] = pd.DatetimeIndex(inputForm['date']).day
    inputForm.drop(['dc_id'], inplace=True, axis=1)
    return inputForm


#将月份和年份都转化成哑变量
def dummies_exer(dummyForm):
    X = dummyForm.loc[:, ['year', 'month']]
    enc = preprocessing.OneHotEncoder(handle_unknown='ignore')
    enc.fit(X)
    dummyForm[['Year_2016', 'Year_2017', 'Jan', 'Feb',
               'Mar', 'Apr', 'May',
               'July', 'Aug', 'Sep',
               'Oct', 'Dec']] = pd.DataFrame(enc.transform(X).toarray())
    return dummyForm





def cate_tree(f):
    f1 = pd.pivot_table(f, index=['item_second_cate_cd', 'year', 'month'], values=['quantity'],
                        columns=['item_third_cate_cd'], aggfunc=[np.mean])
    f1.query('item_second_cate_cd == 29 and year == 2016')





def pivot_table(f):
    uni_secondNum = list(f['item_second_cate_cd'].unique())
    for i in uni_secondNum:
        f1 = pd.pivot_table(f, index=['item_second_cate_cd', 'month'], values=['quantity', 'discount'],
                            columns='year', aggfunc={'quantity': np.mean, 'discount': np.mean})
        print(f1)
    # f1 = pd.pivot_table(f, index=['item_second_cate_cd', 'year', 'month'], values=['quantity', 'discount'],
    #                     aggfunc='mean')
    # f2 = f1.query('item_second_cate_cd').reset_index(['item_second_cate_cd', 'year', 'month']).set_index('month')
    # k = 1
    # for i in uni_secondNum:
    #     col1 = f2[(f2['item_second_cate_cd'] == i) & (f2['year'] == 2016)]['quantity']
    #     col2 = f2[(f2['item_second_cate_cd'] == i) & (f2['year'] == 2017)]['quantity']
    #     plt.subplot(3, 3, k)
    #     l1, =plt.plot(col1)
    #     l2, =plt.plot(col2)
    #     plt.legend(handles=[l1, l2], labels=['2016', '2017'], loc='best')
    #     plt.xlabel('month')
    #     plt.ylabel('avr month sales')
    #     plt.title('sec%d'%i)
    #     k+=1
    # plt.tight_layout()
    # plt.show()

def drawCumulativeHist(f):
    uni_secondNum = list(f['item_second_cate_cd'].unique())
    f1 = pd.pivot_table(f, index=['item_second_cate_cd'], values=['quantity'],
                        aggfunc='mean')
    f2 = f1.query('item_second_cate_cd').reset_index(['item_second_cate_cd'])
    print(f2)

    # k = 1
    # for i in uni_secondNum:
    #     for j in [2016, 2017]:
    #         col = f2[(f2['item_second_cate_cd'] == i) & (f2['year'] == j)]['quantity'].cumsum()
    #         plt.subplot(4, 4, k)
    #         plt.plot(col)
    #         plt.title("sec%d in Year%d"%(i, j))
    #         k+=1
    # plt.tight_layout()
    # plt.show()

    #散点图表示出折扣和销量的关系
    # k = 1
    # for i in uni_secondNum:
    #     for j in [2016, 2017]:
    #         col1 = f2[(f2['item_second_cate_cd'] == i) & (f2['year'] == j)]['quantity']
    #         col2 = f2[(f2['item_second_cate_cd'] == i) & (f2['year'] == j)]['discount']
    #         plt.subplot(4, 4, k)
    #         plt.scatter(col1, col2)
    #         plt.title("sec%d in Year%d"%(i, j))
    #         k+=1
    # plt.tight_layout()
    # plt.show()






def day_sales(f):
    uni_secondNum = list(f['item_second_cate_cd'].unique())
    f1 = pd.pivot_table(f, index=['item_second_cate_cd','year', 'month', 'day'], values=['quantity'],
                        aggfunc='mean')
    f2 = f1.query('item_second_cate_cd').reset_index(['item_second_cate_cd', 'year', 'month', 'day']).set_index('day')
    k = 1
    for i in uni_secondNum:
        for j in np.arange(1, 4):
            col1 = f2[(f2['item_second_cate_cd'] == i) & (f2['year'] == 2016) & (f2['month'] == j)]['quantity']
            col2 = f2[(f2['item_second_cate_cd'] == i) & (f2['year'] == 2017) & (f2['month'] == j)]['quantity']
            plt.subplot(4, 4, k)
            l1, = plt.plot(col1)
            l2, = plt.plot(col2)
            # plt.legend(handles=[l1, l2], labels=['2016', '2017'], loc='best')
            plt.xlabel('day')
            plt.ylabel('avr day sales')
            plt.title('sec%d month%d' % (i, j))
            k += 1

    plt.tight_layout()
    plt.show()



def gaussian_reg(df):
    df = df.iloc[:1000, :]
    X = df[['promotion_type1', 'promotion_type4', 'promotion_type6', 'promotion_type10', 'date_month']]
    y = df['quantity']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)

    gp.fit(X_train, y_train)
    print(gp.score(X_test, y_test))

ff = readForm(1, 5)
print(ff['item_second_cate_cd'].value_counts())
print(ff['item_third_cate_cd'].value_counts())



#
# df_train = readForm(7, 0)
# print(df_train.shape, df_train.head(2), df_train.dtypes)
# categoryList = ['item_second_cate_cd', 'item_third_cate_cd', 'brand_code', 'year', 'month', 'day']
# for var in categoryList:
#     df_train[var] = df_train[var].astype('category')
#
# # msno.matrix(df_train, labels=True)
#
# print("original datasets quantity distribution:\n")
# print(df_train['quantity'].describe())
# quan = df_train['quantity']
#
#
#
# #计算最小估计值和最大估计值
# minProb = np.percentile(quan, 25) - 1.5 * (np.percentile(quan, 75) - np.percentile(quan, 25))
# maxProb = np.percentile(quan, 75) + 1.5 * (np.percentile(quan, 75) - np.percentile(quan, 25))
#
# df_train = df_train[(quan > minProb) & (quan < maxProb)]
# # df_train = df_train[np.abs(df_train["quantity"]-df_train["quantity"].mean())<=(3*df_train["quantity"].std())]
# print(df_train['quantity'].describe())
#
#
#
# sn.set(style='whitegrid')
# fig, axes = plt.subplots(nrows=2, ncols=2)
# sn.distplot((df_train["quantity"]), ax=axes[0][0])
# sn.boxplot(data=df_train, y="quantity", x="year", orient="v", whis = 1.5, ax=axes[0][1])
# sn.boxplot(data=df_train,y="quantity", x="month",orient="v", whis=1.5, ax=axes[1][0])
# sn.boxplot(data=df_train,y="quantity",x="item_second_cate_cd",orient="v", whis=1.5, ax=axes[1][1])
#
# axes[0][0].set(ylabel='quantity',title="Box Plot On quantity")
# axes[0][1].set(xlabel='Year',  title="Box Plot On Count Across Season")
# axes[1][0].set(xlabel='Month', title="Box Plot On Count Across Hour Of The Day")
# axes[1][1].set(xlabel='Second',  title="Box Plot On Count Across Working Day")
#
#
#
# plt.show()






#关于时间序列平稳性检验的两个函数
# def item42_ADF(ts):
#     from statsmodels.tsa.stattools import adfuller
#     print("results of the DF Test:")
#     dftest = adfuller(ts, autolag='AIC')
#     output = pd.Series(dftest[0:4], index=['Test Statistic', 'p_value', '#lags Used', 'number of observations'])
#     for key, value in dftest[4].items():
#         output['Critical Value(%s)'%key] = value
#     print(output)
#
# def item42_KPSS(ts):
#     from statsmodels.tsa.stattools import kpss
#     print("results of kpss Test:")
#     kpsstest = kpss(ts, regression='c')
#     output = pd.Series(kpsstest[0:3], index=['Test Statistic', 'p_value', 'lags used'])
#     for key, value in kpsstest[3].items():
#         output['Critical Value(%s)'%key] = value
#     print(output)






# # c42 = c6d0[c6d0['item_sku_id'] == 42]
# c90 = c6d0[c6d0['item_sku_id'] == 90]
# # c176 = c6d0[c6d0['item_sku_id'] == 176]
# # c187 = c6d0[c6d0['item_sku_id'] == 187]
# c90.set_index('date', inplace=True)
# print()
#
# # plt.plot(c42['date'], c42['quantity'])
# plt.plot(c90['2016'].loc[:,'quantity'])
# # plt.plot(c176['date'], c176['quantity'])
# # plt.plot(c187['date'], c187['quantity'])
# plt.xticks(rotation=80)
# plt.ylim(0, 10)
# plt.show()





#从属性表格中把第一类目==1的产品挑选出来，存到attr_first_cate==1的表格里
# attr_dataframe = pd.read_csv("/Users/mac/文档/京东论文/forecast_data_0523/sku_attr.csv")
# info2 = pd.merge(first_cate1, attr_dataframe)
# info2.to_csv("/Users/mac/文档/京东论文/forecast_data_0523/attr_first_cate==1.csv")

#从促销表格中把第一类目==1的产品挑选出来，存到prom_first_cate==1的表格里
# prom_df = pd.read_csv("/Users/mac/文档/京东论文/forecast_data_0523/sku_prom.csv")
# info3 = pd.merge(first_cate1, prom_df)
# info3.to_csv("/Users/mac/文档/京东论文/forecast_data_0523/prom_first_cate==1.csv")

#从促销和销量的整合表格里提取第一类目==1的促销和销量的平行坐标轴
# dc0 = pd.read_csv("/Users/mac/文档/京东论文/forecast_data_0523/dc0.csv")
# info4 = pd.merge(first_cate1, dc0)
# info4.drop(["item_first_cate_cd", "item_second_cate_cd", "item_third_cate_cd", "brand_code"], axis=1, inplace = True)
# info4.to_csv("/Users/mac/文档/京东论文/forecast_data_0523/prom_quantity_first_cate==1.csv")

# dateparse = lambda dates: pd.datetime.strptime(dates, "%Y/%m/%d")
# quantity_accumulative = pd.read_csv("/Users/mac/文档/京东论文/forecast_data_0523/prom_quantity_first_cate==1.csv",
#                                     date_parser=dateparse)
# quantity_accumulative['date'] = pd.to_datetime(quantity_accumulative["date"])
# copy_one = quantity_accumulative.copy()
# copy_one = copy_one.set_index('date')
# g1 = quantity_accumulative.groupby(['item_sku_id', 'date']).sum()
# print(g1)

#构建多层次索引,用来看月销量的累计和
# date_column = quantity_accumulative.iloc[:, 1]
# id_column = quantity_accumulative.iloc[:, 0]
# data_reshape = pd.DataFrame({'promotion_type1': list(quantity_accumulative["promotion_type1"]),
#                              'promotion_type4':list(quantity_accumulative["promotion_type4"]),
#                              'promotion_type6': list(quantity_accumulative["promotion_type6"]),
#                              'promotion_type10': list(quantity_accumulative["promotion_type10"]),
#                              'quantity': list(quantity_accumulative["quantity"])},
#                             index=[id_column, date_column])
# data_reshape.index.names=['row1', 'row2']


# def get_item_date(df):
#     new_dict = {}
#     for i in np.arange():
#
#
#
#     temp1 = df["row1"]

# print(data_reshape.sum(level='row2')['quantity'].plot())
# plt.show()
# print(quantity_accumulative.groupby(["item_sku_id"], ["2016/01"])["quantity"].sum())

# sku_quantity_accu = quantity_accumulative.groupby("item_sku_id")["quantity"].sum()
# print(sku_quantity_accu)
