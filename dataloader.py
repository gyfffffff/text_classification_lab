from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import json
import pickle
from tqdm import tqdm

class dataloader(object):
    def __init__(self, method='tf-idf'):
        self.train_data_path = r'exp1_data\train_data.txt'
        self.puncs = [',','.','(',')',';',':','[',']','{','}']
        self.labels = []
        self.method = method

    def get_raw(self):
        corpus_raw = []
        labels = []
        with open(self.train_data_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                corpus_raw.append(json.loads(line)['raw'])
                labels.append(json.loads(line)['label'])
        return corpus_raw, labels
    
    def remove_stopwords(self):
        corpus_raw, self.labels = self.get_raw()

        # 去除停用词和标点
        corpus = []
        print('remove stopwords...')
        for line in tqdm(corpus_raw):
            filtered_line = []
            words = line.split(' ')
            for word in words:
                if word not in ENGLISH_STOP_WORDS:
                    for p in self.puncs:
                        word = word.strip(p)
                    filtered_line.append(word.lower())
            corpus.append(' '.join(filtered_line))
        return corpus

    def split_dataset(self):
        corpus = self.remove_stopwords()
        X_train, X_test, y_train, y_test = train_test_split(
            corpus, self.labels, test_size=0.2, random_state=0)
        
        pickle.dump(X_train, open('X_train.pkl', 'wb'))
        pickle.dump(X_test, open('X_test.pkl', 'wb'))
        pickle.dump(y_train, open('y_train.pkl', 'wb'))
        pickle.dump(y_test, open('y_test.pkl', 'wb'))

        print('\n split dataset successfully!')

    def load_raw_traindata(self):
        return pickle.load(open('X_train.pkl', 'rb')), pickle.load(open('y_train.pkl', 'rb'))
    
    def load_raw_testdata(self):
        return pickle.load(open('X_test.pkl', 'rb')), pickle.load(open('y_test.pkl', 'rb'))
    
    def vectorize(self, X):
        if self.method == 'tf-idf':
            X_train = self.load_raw_traindata()[0]

            # 统计出现次数而不考虑顺序
            bigram_vectorizer = CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', min_df=1) # 会提取至少两个字母的单词
            bigram_vectorizer.build_analyzer()
            bigram_vectorizer.fit(X_train)
            X_2 = bigram_vectorizer.transform(X) #.toarray()

            # tf-idf
            transformer = TfidfTransformer(smooth_idf=True)
            transformer.fit_transform(X_2)

            print('vectorize finished! shape: {}\n'.format(X_2.shape))

            return X_2
        
        else:
            print('not support method: {}'.format(self.method))
            return None

    
    def load_traindata(self):
        return self.vectorize(self.load_raw_traindata()[0]), self.load_raw_traindata()[1]
    def load_testdata(self):
        return self.vectorize(self.load_raw_testdata()[0]), self.load_raw_testdata()[1]
    
if __name__ == '__main__':
    dl = dataloader()
    dl.split_dataset()
    
