import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer
import json

# data_path = r'exp1_data'
train_data_path = r'exp1_data\train_data.txt'
# train_raw = pd.read_csv(train_data_path)
# print(train_row)

vectorizer = CountVectorizer()
corpus = []
with open(train_data_path, 'r') as f:
    lines = f.readlines()
    for line in lines:
        corpus.append(json.loads(line)['raw'])

X = vectorizer.fit_transform(corpus)
analyze = vectorizer.build_analyzer() # 会提取至少两个字母的单词
# ['this', 'is', 'text', 'document', 'to', 'analyze']
# print(analyze("This is a text document to analyze."))

print(vectorizer.get_feature_names())
