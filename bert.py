from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from tqdm import tqdm

def bert(X):
    
    checkpoint = "bert-base-uncased"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModel.from_pretrained(checkpoint).to(device)

    X_2 = []

    print('bert vectorizing...')
    for text in tqdm(X):
        tokens_tensor = tokenizer.encode_plus(
            text, add_special_tokens=True, return_tensors="pt"  # 添加[CLS]和[SEP]标记
        )['input_ids'].to(device)
        with torch.no_grad():
            outputs = model(tokens_tensor)
            hidden_states = outputs.pooler_output
        # print(hidden_states)
        X_2.append(hidden_states[0,:].tolist())
    
    return np.array(X_2)

if __name__ == '__main__':
    bert()
