from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import json

train_data_path = r'exp1_data\train_data.txt'

corpus_raw = []
labels = []
with open(train_data_path, 'r') as f:
    lines = f.readlines()
    for line in lines:
        corpus_raw.append(json.loads(line)['raw'])
        labels.append(json.loads(line)['label'])

# 去除停用词和标点
puncs = [',','.','(',')',';',':','[',']','{','}']

corpus = []
for line in corpus_raw:
    filtered_line = []
    words = line.split(' ')
    for word in words:
        if word not in ENGLISH_STOP_WORDS:
            for p in puncs:
                word = word.strip(p)
            filtered_line.append(word.lower())
    corpus.append(' '.join(filtered_line))

# print(corpus[1])

X_train, X_test, y_train, y_test = train_test_split(
    corpus, labels, test_size=0.2, random_state=0)

bigram_vectorizer = CountVectorizer(ngram_range=(1, 2),
                                    token_pattern=r'\b\w+\b', min_df=1) # 会提取至少两个字母的单词
analyze = bigram_vectorizer.build_analyzer()
X_2 = bigram_vectorizer.fit_transform(corpus)
# print(X_2)

# tf-idf
transformer = TfidfTransformer(smooth_idf=False)
tfidf = transformer.fit_transform(X_2)
# print(tfidf.toarra()[0,:])
