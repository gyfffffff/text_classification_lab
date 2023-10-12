import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from torch.optim import Adam
from tqdm import tqdm
import logging
import pickle
from sklearn.model_selection import train_test_split
from dataloader import dataloader
import numpy as np


class Dataset(torch.utils.data.Dataset):
    def __init__(self, corpus, y):
        self.labels = y
        self.texts = [
            tokenizer(
                text,
                padding="max_length",
                max_length=128,
                truncation=True,
                return_tensors="pt",
            )
            for text in corpus
        ]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):
        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)
        return batch_texts, batch_y


class BertClassifier(nn.Module):
    def __init__(self, dropout=0.1):
        super(BertClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained("bert-base-cased")
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 10)

    def forward(self, input_id, mask):
        # 第一个变量 _包含sequence中所有 token 的 Embedding 向量层。
        # 第二个变量pooled_output包含 [CLS] token 的 Embedding 向量。
        _, pooled_output = self.bert(
            input_ids=input_id, attention_mask=mask, return_dict=False
        )
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        # final_layer = self.linear2(self.relu(linear_output))
        return linear_output


def train(model, train_data, train_y, val_data, val_y, learning_rate, epochs):
    # 通过Dataset类获取训练和验证集
    train, val = Dataset(train_data, train_y), Dataset(val_data, val_y)
    # DataLoader根据batch_size获取数据，训练时选择打乱样本
    train_dataloader = torch.utils.data.DataLoader(train, batch_size=32, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=32, shuffle=True)
    # 判断是否使用GPU
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9999)

    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()
    # 开始进入训练循环
    bestmodel = None
    bestvalacc = 0
    for epoch_num in range(epochs):
        # 定义两个变量，用于存储训练集的准确率和损失
        total_acc_train = 0
        total_loss_train = 0
        # 进度条函数tqdm
        for train_input, train_label in tqdm(train_dataloader):
            train_label = train_label.to(device)
            mask = train_input["attention_mask"].to(device)
            input_id = train_input["input_ids"].squeeze(1).to(device)
            # 通过模型得到输出
            output = model(input_id, mask)
            # 计算损失
            batch_loss = criterion(output, train_label)
            total_loss_train += batch_loss.item()
            # 计算精度
            acc = (output.argmax(dim=1) == train_label).sum().item()
            total_acc_train += acc
            # 模型更新
            model.zero_grad()
            batch_loss.backward()
            optimizer.step()
            scheduler.step()
        # ------ 验证模型 -----------
        # 定义两个变量，用于存储验证集的准确率和损失
        total_acc_val = 0
        total_loss_val = 0
        # 不需要计算梯度
        with torch.no_grad():
            # 循环获取数据集，并用训练好的模型进行验证
            for val_input, val_label in val_dataloader:
                # 如果有GPU，则使用GPU，接下来的操作同训练
                val_label = val_label.to(device)
                mask = val_input["attention_mask"].to(device)
                input_id = val_input["input_ids"].squeeze(1).to(device)

                output = model(input_id, mask)

                batch_loss = criterion(output, val_label)
                total_loss_val += batch_loss.item()

                acc = (output.argmax(dim=1) == val_label).sum().item()
                total_acc_val += acc

        logging.info(
            f"""Epochs: {epoch_num + 1} 
              | Train Loss: {total_loss_train / len(train_data): .3f} 
              | Train Accuracy: {total_acc_train / len(train_data): .3f} 
              | Val Loss: {total_loss_val / len(val_data): .3f} 
              | Val Accuracy: {total_acc_val / len(val_data): .3f}"""
        )

        if bestvalacc < total_acc_val / len(val_data):
            bestvalacc = total_acc_val / len(val_data)
            bestmodel = model
    pickle.dump(bestmodel, open(f"model/{modelname}.pkl", "wb"))
    logging.info(f"model saved in model/{modelname}.pkl")


if __name__ == "__main__":
    corpus_raw, labels = dataloader().get_raw()
    # corpus_raw = corpus_raw[:2000]
    # labels = labels[:2000]
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    modelname = "bert"
    logging.basicConfig(
        format="%(asctime)s %(message)s",
        filename="log/{}.txt".format(modelname),
        filemode="a+",
        level=logging.INFO,
    )

    corpus_train, corpus_val, y_train, y_val = train_test_split(
        corpus_raw, labels, test_size=0.15, random_state=2023
    )

    EPOCHS = 5
    model = BertClassifier()
    LR = 1e-5
    train(model, corpus_train, y_train, corpus_val, y_val, LR, EPOCHS)
