import torch
from torch import nn
from tqdm import tqdm
import logging
import pickle

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(310271, 1024)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        return x

def mlp(x_train, y_train, device, modelname, train):

    logging.basicConfig(format='%(asctime)s %(message)s',
            filename='log/{}.txt'.format(modelname),
            filemode='a+',
            level=logging.INFO)
    if train == True:

        model = MLP().to(device)

        # 损失函数
        loss_fn = nn.CrossEntropyLoss().to(device)

        # 优化器
        learning_rate = 1e-2
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

        # 设置训练网络的参数
        total_train_step = 0  # 记录训练的次数
        epoch = 100   # 训练的轮数

        batchsize = 64
        
        bestmodle = None
        bestacc = 0
        
        earlystop_a = 0.02  # 早停阈值，泛华损失超过该阈值，就提前停止训练
        minloss = 10000
        

        x_train = x_train.tocoo()
        x_train = torch.sparse.FloatTensor(
            torch.LongTensor([x_train.row.tolist(),x_train.col.tolist()]), torch.Tensor(x_train.data)
        ).to_dense()
        y_train = torch.Tensor(y_train).long()
        
        # shuffle
        idx = torch.randperm(x_train.shape[0])
        x_train = x_train[idx, :].view(x_train.size())
        y_train = y_train[idx].view(y_train.size())
        
        # 划分出验证集
        val_n = 2*batchsize
        x_val = x_train[:val_n, :].to(device)
        y_val = y_train[:val_n].to(device)
        
        x_train = x_train[val_n:, :].to(device)
        y_train = y_train[val_n:].to(device)
        
        n_samples = x_train.shape[0]
        
        for i in range(epoch):
            torch.cuda.empty_cache()
            logging.info("===================第{}轮开始=======================\n".format(i+1))
            for j in range(0, n_samples, batchsize):
                x_batch = x_train[j:j+batchsize, :].to(device)
                y_batch = y_train[j:j+batchsize].to(device)
                predict = model(x_batch)
                loss = loss_fn(predict, y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_train_step += 1
                if total_train_step % 10 == 0:
                    logging.info("第{}轮，第{}批, loss: {}".format(i+1, j/batchsize, loss.item()))

            total_val_loss = 0
            total_val_accuracy = 0
            with torch.no_grad():
                predict = model(x_val)
                loss = loss_fn(predict, y_val)
                total_val_loss += loss.item()
                accuracy = (predict.argmax(1) == y_val).sum()
                total_val_accuracy += accuracy
            logging.info("\n第{}轮, 验证集上loss: {}, accuracy: {}\n".format(i+1, total_val_loss, total_val_accuracy/val_n))
            if bestacc < total_val_accuracy/val_n:
                bestacc = total_val_accuracy/val_n
                bestmodel = model
            if minloss > total_val_loss:
                minloss = total_val_loss
            
            # 早停策略
            gen_loss = total_val_loss/minloss - 1
            if gen_loss >= earlystop_a:
                logging.info('Apply earlystop.')
                break
        
        pickle.dump(model, open(f'model/{modelname}.pkl', 'wb'))
        logging.info(f'model saved in model/{modelname}.pkl')
            

        


        


        