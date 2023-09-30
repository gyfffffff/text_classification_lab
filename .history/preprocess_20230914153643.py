import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import json

train_data_path = r'exp1_data\train_data.txt'

corpus_raw = []
with open(train_data_path, 'r') as f:
    lines = f.readlines()
    for line in lines:
        corpus_raw.append(json.loads(line)['raw'])

# 去除停用词
# ENGLISH_STOP_WORDS = list(ENGLISH_STOP_WORDS)
puncs = [',','.','(',')',';',':','[',']','{','}']
# print(ENGLISH_STOP_WORDS)

corpus = []
for line in corpus_raw:
    filtered_line = []
    words = line.split(' ')
    for word in words:
        if word not in ENGLISH_STOP_WORDS:
            filtered_line.append(word)
        for p in puncs:
            if p in filtered_line[-1]:
                idx = filtered_line[-1].index(p)
                filtered_line[-1] = filtered_line[-1][:p] + fi
    corpus.append(filtered_line)

print(corpus[0])

# bigram_vectorizer = CountVectorizer(ngram_range=(1, 2),
#                                     token_pattern=r'\b\w+\b', min_df=1) # 会提取至少两个字母的单词
# analyze = bigram_vectorizer.build_analyzer()


# X_2 = bigram_vectorizer.fit_transform(corpus).toarray()

# print(X_2)

