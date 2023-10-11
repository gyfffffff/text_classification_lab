# 本代码是为了使用所有数据，使用最优模型和参数，训练用于提交版本的模型。

from dataloader import dataloader
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
from mlp import mlp
import pickle
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from tqdm import tqdm

X_test, y_test = dataloader().load_raw_testdata()
X_train, y_train = dataloader().load_raw_traindata()



X_train = X_train + X_test
y_train = y_train + y_test
    
tv = TfidfVectorizer(stop_words='english')
tv.fit(X_train)

X_train = tv.transform(X_train)

def get_raw(path):
    corpus_raw = []
    
    with open(path, 'r') as f:
        f.readline()
        lines = f.readlines()
        for line in lines:
            corpus_raw.append(','.join(line.split(',')[1:]).strip())
    return corpus_raw  
def remove_stopwords(corpus):
    corpus = []
    print("remove stopwords...")
    for line in tqdm(corpus_raw):
        filtered_line = []
        words = line.split(" ")
        for word in words:
            if word not in ENGLISH_STOP_WORDS:
                for p in [",", ".", "(", ")", ";", ":", "[", "]", "{", "}"]:
                    word = word.strip(p)
                filtered_line.append(word.lower())
        corpus.append(" ".join(filtered_line))
    return corpus
    
corpus_raw= get_raw('exp1_data/test.txt')
# print(corpus_raw)
corpus = remove_stopwords(corpus_raw)
X_test = tv.transform(corpus)

print(X_train.shape)
print(X_test.shape)

# mlp(X_train, y_train, 'cuda', 'mlp_submit_final', 1, 'tf-idf')

model = pickle.load(open('model/mlp_submit_final.pkl', 'rb'))
X_test = X_test.tocoo()
X_test = torch.sparse_coo_tensor(
    [X_test.row.tolist(), X_test.col.tolist()],
    torch.Tensor(X_test.data),
    size=X_test.shape,
).to_dense().to('cuda')
y_test = torch.Tensor(y_test).long().to('cuda')
preds = model(X_test).argmax(1)


with open('submit.txt', 'w') as f:
    f.write('id, pred\n')
    for _id, pred in enumerate(preds):
        f.write(f'{_id}, {pred.item()}\n')
