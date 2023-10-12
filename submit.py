# 本代码是为了使用所有数据，使用最优模型和参数，训练用于提交版本的模型。

from dataloader import dataloader
from transformers import AutoTokenizer
import pickle


def get_raw(path):
    corpus_raw = []

    with open(path, "r") as f:
        f.readline()
        lines = f.readlines()
        for line in lines:
            corpus_raw.append(",".join(line.split(",")[1:]).strip())
    return corpus_raw


tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
device = "cuda"
corpus_raw = get_raw("exp1_data/test.txt")
model = pickle.load(open("model/bert.pkl", "rb"))


with open("result.txt", "w") as f:
    f.write("id, pred\n")

    for _id, text in enumerate(corpus_raw):
        output = tokenizer(
            text,
            padding="max_length",
            max_length=120,
            truncation=True,
            return_tensors="pt",
        ).to(device)
        pred = model(output["input_ids"], output["attention_mask"]).argmax().item()
        f.write(f"{_id}, {pred}\n")
