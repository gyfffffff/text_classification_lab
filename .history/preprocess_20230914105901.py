import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer
import json

# data_path = r'exp1_data'
train_data_path = r'exp1_data\train_data.txt'
# train_row = pd.read_csv(train_data_path)
# print(train_row)

vectorizer = CountVectorizer()
with open(train_data_path, 'r') as f:
    print(f.readline())
    print()

