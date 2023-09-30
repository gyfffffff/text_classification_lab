import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer

# data_path = r'exp1_data'
train_data_path = r'exp1_data\train_data.txt'
# train_row = pd.read_csv(train_data_path)
# print(train_row)

vectorizer = CountVectorizer()
print(vectorizer
