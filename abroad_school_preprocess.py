import numpy as np
import pandas as pd
import string
import requests
from bs4 import BeautifulSoup


# #导入出国的信息
# f1 = pd.read_csv("/Users/mac/Desktop/original.csv", usecols=["学号", "就业形式", "具体落实单位名称"])
# f2 = f1[f1["就业形式"] == '已出国']
# f2.columns = ["学号", "就业形式", "出国去往学校"]


#网页上爬虫导入2019QS排名
response = requests.get("https://www.forwardpathway.com/28769")
soup = BeautifulSoup(response.text, 'lxml')
tables = soup.select('table')
df_list = []
for table in tables:
    df_list.append(pd.concat(pd.read_html(table.prettify())))
f3 = pd.concat(df_list)
f3.rename(columns={'学校中文名': '出国去往学校'}, inplace=True)
f3.to_csv("/Users/mac/Desktop/QS_rank.csv", encoding="utf_8_sig")
correct_name_multi = list(f3["出国去往学校"])
correct_name = list(set(correct_name_multi))

#
# #开始比对问卷表名称和QS表名称
# new_list = []
# for i in np.arange(0, len(f2)):
#     flag = 0
#     for j in np.arange(0, len(correct_name)):
#         if (f2.iloc[i, 2].find(correct_name[j])) != -1:
#             new_item = f2.iloc[i, 2].replace(f2.iloc[i, 2], correct_name[j])
#             new_list.append(new_item)
#             flag = 1
#             break
#         elif (correct_name[j].find(f2.iloc[i, 2])) != -1:
#             item_new = f2.iloc[i, 2].replace(f2.iloc[i, 2], correct_name[j])
#             new_list.append(item_new)
#             flag = 1
#             break
#     if flag == 0:
#         new_list.append(f2.iloc[i, 2])
#
# f2['出国去往学校'] = new_list
# f4 = pd.merge(f2, f3, how='left', on='出国去往学校')
#
# f4.to_csv("/Users/mac/Desktop/QS_schoolnumber.csv", encoding="utf_8_sig")

f5 = pd.read_excel("/Users/mac/Desktop/school_abroad.xls", usecols=["学号", "就业形式", "具体落实单位名称"])
f2= f5[f5["就业形式"] == "已出国"]
f2.columns = ["学号","就业形式", "出国去往学校"]
new_list = []
for i in np.arange(0, len(f2)):
    flag = 0
    for j in np.arange(0, len(correct_name)):
        if (f2.iloc[i, 2].find(correct_name[j])) != -1:
            new_item = f2.iloc[i, 2].replace(f2.iloc[i, 2], correct_name[j])
            new_list.append(new_item)
            flag = 1
            break
        elif (correct_name[j].find(f2.iloc[i, 2])) != -1:
            item_new = f2.iloc[i, 2].replace(f2.iloc[i, 2], correct_name[j])
            new_list.append(item_new)
            flag = 1
            break
    if flag == 0:
        new_list.append(f2.iloc[i, 2])

f2['出国去往学校'] = new_list
f4 = pd.merge(f2, f3, how='left', on='出国去往学校')
f4.to_csv("/Users/mac/Desktop/abroad.csv", encoding="utf_8_sig")
















