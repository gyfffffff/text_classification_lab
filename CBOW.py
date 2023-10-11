import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange, tqdm
import pickle
import numpy as np


learning_rate = 0.001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
context_size = 2


class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.proj = nn.Linear(embedding_dim, 128)
        self.output = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = sum(self.embeddings(inputs)).view(1, -1)
        out = F.relu(self.proj(embeds))
        out = self.output(out)
        nll_prob = F.log_softmax(out, dim=-1)
        return nll_prob



def make_context_vector(context, word_to_ix):
    idxs = [word_to_ix[w] for w in context]
    return torch.tensor(idxs, dtype=torch.long)

def train_cbow(texts):  # texts: [text1, text2, text3...]
    embedding_dim = 100
    epoch = 10
    wordList = []
    for _text in texts:
        text = _text.split(' ')
        for word in text:
            if word not in wordList:
                wordList.append(word)
    # print("wordList=", wordList)

    vocab_size = len(wordList)

    word_to_idx = {word: i for i, word in enumerate(wordList)}
    idx_to_word = {i: word for i, word in enumerate(wordList)}

    # cbow词表，{[w1,w2,w4,w5],"label"}
    data = []
    for i in range(2, len(wordList) - 2):
        context = [wordList[i - 2], wordList[i - 1],
                   wordList[i + 1], wordList[i + 2]]
        target = wordList[i]
        data.append((context, target))
    
    model = CBOW(vocab_size, embedding_dim).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    losses = []
    loss_function = nn.NLLLoss()

    for epoch in trange(epoch):
        total_loss = 0
        for context, target in tqdm(data):
            context_vector = make_context_vector(
                context, word_to_idx).to(device) 
            target = torch.tensor([word_to_idx[target]]).cuda() 
            model.zero_grad()
            train_predict = model(context_vector).cuda() 
            loss = loss_function(train_predict, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        losses.append(total_loss)

        print(f'epoch {epoch}: loss {total_loss}')

    W = model.embeddings.weight.cpu().detach().numpy()

    # 词嵌入字典，即{单词1:词向量1,单词2:词向量2...}的格式
    word_2_vec = {}
    for word in word_to_idx.keys():
        word_2_vec[word] = W[word_to_idx[word], :]

    with open("exp1_data/CBOW_wordvec100.txt", 'w', encoding='utf-8') as f:
        for key in word_to_idx.keys():
            f.writelines(str(key)+' '+ ' '.join([str(i) for i in word_2_vec[key]]))
            f.writelines('\n')

def cbow(corpus):
    word2vec = open("exp1_data/CBOW_wordvec100.txt", 'r', encoding='utf-8').readlines()
    word2vec_dict = {wv.split(' ')[0]: wv.split(' ')[1:] for wv in word2vec}

    vectorized_corpus = []

    print('Vectorizing...')

    for sentence in tqdm(corpus):
        vec = [0] * (len(word2vec[0].split(' '))-1)
        count = 0

    
        for word in sentence.split(' '):
            if word in word2vec_dict:
                vec = [x + float(y) for x, y in zip(vec, word2vec_dict[word])]
                count += 1

        if count != 0:
            vec = [x / count for x in vec]

        vectorized_corpus.append(vec)

    return np.array(vectorized_corpus)

if '__main__' == __name__:
    X_train = pickle.load(open("X_train.pkl", 'rb'))
    train_cbow(X_train)