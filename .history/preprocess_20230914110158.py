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
        corpus.append(line['raw'])

X = vectorizer.fit_transform(corpus)

