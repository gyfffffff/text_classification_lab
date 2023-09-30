import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer
import json

train_data_path = r'exp1_data\train_data.txt'

corpus = []
with open(train_data_path, 'r') as f:
    lines = f.readlines()
    for line in lines:
        corpus.append(json.loads(line)['raw'])

# 去除停用词
stop_words_path = r'exp1_data\stopwords.txt'
for line in corpus:
    



# bigram_vectorizer = CountVectorizer(ngram_range=(1, 2),
#                                     token_pattern=r'\b\w+\b', min_df=1) # 会提取至少两个字母的单词
# analyze = bigram_vectorizer.build_analyzer()


# X_2 = bigram_vectorizer.fit_transform(corpus).toarray()

# print(X_2)
