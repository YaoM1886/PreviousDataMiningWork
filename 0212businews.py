import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import jieba
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE








#read data
df = pd.read_csv("/Users/mac/Desktop/business.csv", iterator=True, sep=';', encoding='utf-8')

df1 = df.get_chunk(3458611)
df1 = df1.sample(frac=0.0005)
column_list = list(df1) #columns names


#转换成时间类型
df1['publish_time'] = pd.to_datetime(df1['NEWS_PUBLISH_TIME'], format='%Y-%m-%dT%H:%M:%S')
df1['publish_year'] = pd.DatetimeIndex(df1['publish_time']).year
df1['publish_month'] = pd.DatetimeIndex(df1['publish_time']).month
df1['publish_day'] = pd.DatetimeIndex(df1['publish_time']).day
df1['publish_hour'] = pd.DatetimeIndex(df1['publish_time']).hour
df1['publish_minute'] = pd.DatetimeIndex(df1['publish_time']).minute
df1['publish_second'] = pd.DatetimeIndex(df1['publish_time']).second


df1['period'] = df1['publish_year'].map(str) + "-"+  df1['publish_month'].map(str)
df1['period'] = pd.to_datetime(df1['period'], format='%Y-%m')

df1['per'] = df1['publish_year'].map(str) + "-"+  df1['publish_month'].map(str) + '-' + df1['publish_day'].map(str)
df1['per'] = pd.to_datetime(df1['per'], format='%Y-%m-%d')



#
# df1['insert_time'] = pd.to_datetime(df1['INSERT_TIME'], format='%Y-%m-%dT%H:%M:%S')
# df1['insert_year'] = pd.DatetimeIndex(df1['insert_time']).year
# df1['insert_month'] = pd.DatetimeIndex(df1['insert_time']).month
# df1['insert_day'] = pd.DatetimeIndex(df1['insert_time']).day
# df1['insert_hour'] = pd.DatetimeIndex(df1['insert_time']).hour
# df1['insert_minute'] = pd.DatetimeIndex(df1['insert_time']).minute
# df1['insert_second'] = pd.DatetimeIndex(df1['insert_time']).second
#
# df1['update_time'] = pd.to_datetime(df1['UPDATE_TIME'], format='%Y-%m-%dT%H:%M:%S')
# df1['update_year'] = pd.DatetimeIndex(df1['update_time']).year
# df1['update_month'] = pd.DatetimeIndex(df1['update_time']).month
# df1['update_day'] = pd.DatetimeIndex(df1['update_time']).day
# df1['update_hour'] = pd.DatetimeIndex(df1['update_time']).hour
# df1['update_minute'] = pd.DatetimeIndex(df1['update_time']).minute
# df1['update_second'] = pd.DatetimeIndex(df1['update_time']).second


#统计新闻来源类型
#plt.rcParams['font.family']=['STFangsong']

# ax = sns.countplot(x='NEWS_ORIGIN_SOURCE', data=df1)
# plt.show()
# df1['NEWS_PUBLISH_SITE'].value_counts(normalize=True).plot(kind='bar', grid=True, figsize=(16, 9))
# plt.show()


#画时间序列图
# df1['quantity'] = 1
# uni_secondNum = list(df1['NEWS_PUBLISH_SITE'].unique())
# f1 = pd.pivot_table(df1, index=['period'], columns=['NEWS_PUBLISH_SITE'], values=['quantity'],
#                         aggfunc='sum')
# f1 = pd.DataFrame(f1)
# f2 = pd.read_csv('/Users/mac/Desktop/period_year.csv')
# f2.set_index('period_date', inplace=True)
#
# # f2.plot(figsize=(20, 10), linewidth=2, fontsize=20)
# # plt.xlabel('Year', fontsize=20)
# # plt.show()
#
#
# f2[['和讯']].plot(figsize=(20,10), linewidth=2, fontsize=20)
# plt.xlabel('Year', fontsize=20)
# plt.show()

#定义和讯／巨灵的时间序列图
def juling(df):
    dfhexun = df[df['NEWS_PUBLISH_SITE'] == '巨灵新闻'].loc[:,['publish_time', 'NEWS_PUBLISH_SITE']]
    dfhexun['quan'] = 1
    hesite = pd.pivot_table(dfhexun, index=['publish_time'], columns=['NEWS_PUBLISH_SITE'], values=['quan'], aggfunc='sum')
    hesite = pd.DataFrame(hesite)
    hesite.to_csv('/Users/mac/Desktop/juling_time.csv')


#
#
# f3.plot(figsize=(20, 10), linewidth=2, fontsize=20)
# plt.xlabel('Year', fontsize=20)
# plt.show()
# f1.to_csv("/Users/mac/Desktop/period_year_cum.csv", encoding='utf_8_sig')

# source_c.to_csv("/Users/mac/Desktop/counts_PUBLISHSITE.csv", header=None, encoding='utf_8_sig')



#处理标题分词停用词


stop_words = []

f = open('/Users/mac/Desktop/stopwords.txt', 'r')
for l in f.readlines():
    stop_words.append(l.replace('\n', ''))

def tokenizer(text):
    text = text.strip()
    seg_list = jieba.cut(text, cut_all=True)
    out_str = ''
    for word in seg_list:
        if word not in stop_words:
            if word != '\t':
                out_str += word
                out_str += " "
    return out_str
#
df1['tokens'] = df1['NEWS_TITLE'].map(lambda d: tokenizer(d))


# for NEWS_TITLE, tokens in zip(df1['NEWS_TITLE'].head(5), df1['tokens'].head(5)):
#     print('title', NEWS_TITLE)
#     print('tokens:', tokens)
#     print()


#高频词统计
# def keywords(category):
#     tokens = df1[df1['NEWS_PUBLISH_SITE'] == category]['tokens']
#     alltokens = []
#     for i in range(0, len(tokens)):
#         str = tokens.iloc[i]
#         str_split = str.split()
#         str_split = [x for x in str_split if len(x)>1]
#         alltokens.extend(str_split)
#     counter = Counter(alltokens)
#     return counter.most_common(10)
#
# for category in set(df1['NEWS_PUBLISH_SITE']):
#     print('publish_site :', category)
#     print('top 10 keywords:', keywords(category))
#     print('---')


#TF-IDF处理以及降维
tokens = df1['tokens']
for i in range(0, len(tokens)):
    str = tokens.iloc[i]
    tokens.iloc[i] = str.split()
corpus = list(tokens.map(lambda tokens: ' '.join(tokens)))
vectorizer = TfidfVectorizer()
vz = vectorizer.fit_transform(corpus)

svd = TruncatedSVD(n_components=50, random_state=0)
svd_tfidf = svd.fit_transform(vz)

print(svd_tfidf.shape)
run = True
if run:
# run this (takes times)
    tsne_model = TSNE(n_components=2, verbose=1, random_state=0, n_iter=255)
    tsne_tfidf = tsne_model.fit_transform(svd_tfidf)
    print(tsne_tfidf.shape)
    tsne_tfidf_df = pd.DataFrame(tsne_tfidf)
    tsne_tfidf_df.columns = ['x', 'y']
    tsne_tfidf_df['category'] = df1['NEWS_PUBLISH_SITE']
    tsne_tfidf_df['description'] = df1['NEWS_TITLE']
    tsne_tfidf_df.to_csv('/Users/mac/Desktop/tsne_tfidf.csv', encoding='utf_8_sig', index=False)
else:
# or import the dataset directly
    tsne_tfidf_df = pd.read_csv('/Users/mac/Desktop/tsne_tfidf.csv')

groups = tsne_tfidf_df.groupby('category')
fig, ax = plt.subplots(figsize=(15, 10))
ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
for name, group in groups:
    ax.plot(group.x, group.y, marker='o', linestyle='', label=name)
ax.legend()
plt.show()
# tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))
# tfidf = pd.DataFrame(columns=['tfidf']).from_dict(dict(tfidf), orient='index')
# tfidf.columns = ['tfidf']
# # tfidf.tfidf.hist(bins=25, figsize=(15,7))
# # plt.show()
#
#
#
#画关键词词云
# def plot_word_cloud(terms):
#     text = terms.index
#     text = ' '.join(list(text))
#     # lower max_font_size
#     wordcloud = WordCloud(max_font_size=40, font_path="/System/Library/fonts/PingFang.ttc").generate(text)
#     plt.figure(figsize=(25, 25))
#     plt.imshow(wordcloud, interpolation="bilinear")
#     plt.axis("off")
#     plt.show()
#
# plot_word_cloud(tfidf.sort_values(by=['tfidf'], ascending=True).head(40))
# plot_word_cloud(tfidf.sort_values(by=['tfidf'], ascending=False).head(40))

