import torch
from torch import nn
from tqdm import tqdm
import logging
import pickle

class MLP(nn.Module):
    def __init__(self, activate_func, hn, p, init_n):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(init_n, hn),
            nn.Dropout(p),
            activate_func,
            nn.Linear(hn, 10)
        )
        self._initialize_weights()  # Xavier, 不执行则为默认的Kaiming

    def forward(self, x):
        x = self.model(x)
        return x
        
    def _initialize_weights(self):
        for m in self.model.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

def mlp(x_train, y_train, device, modelname, train, vectorize_method):

    learning_rate = 0.25 # 1, 0.1, 0.01, 0.001
    if vectorize_method == 'tf-idf':
        hn = 600 # 3060 2048 1024 
        init_n = 29697
    elif vectorize_method == 'bert':
        hn = 320
        init_n = 768
    else:
        hn = 64
        init_n = 100
    activate_func = nn.PReLU() # nn.ReLU(), nn.LeakyReLU(), nn.PReLU(), nn.tanh(), nn.sigmoid()
    p = 0.3 # 0.15 0.5 0.7
    wdc = 0.001 # 0.01 0.001 0.1
    optim = torch.optim.SGD # torch.optim.Adam
    batchsize = 64 # 64 

    patient = 32  # 有32次验证集上损失增加，就认为模型有过拟合倾向
    patienti = 0

    logging.basicConfig(format='%(asctime)s %(message)s',
            filename='log/{}.txt'.format(modelname),
            filemode='a+',
            level=logging.INFO)
    
    if train == True:
        logging.info(f'lr: {learning_rate}, hn: {hn}, p: {p}, wdc: {wdc}, batchsize: {batchsize},\n \t\t\t\t\t\t relu, sgd, kaiming, cos lr scheduler')
        model = MLP(activate_func, hn, p, init_n).to(device)

        # 损失函数
        loss_fn = nn.CrossEntropyLoss().to(device)

        # 优化器
        optimizer = optim(model.parameters(), lr=learning_rate, weight_decay=wdc)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150, eta_min=1e-4)
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
        #             milestones=[10, 20, 35, 55], gamma=0.99)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9999)

        # 设置训练网络的参数
        total_train_step = 0  # 记录训练的次数
        epoch = 300   # 训练的轮数
        
        bestmodle = None
        bestacc = 0
        
        earlystop_a = 0.0002  # 早停阈值，泛化损失超过该阈值，就提前停止训练
        minloss = 10000

        if vectorize_method == 'tf-idf':
            x_train = x_train.tocoo()
            x_train = torch.sparse.FloatTensor(
                torch.LongTensor([x_train.row.tolist(),x_train.col.tolist()]), torch.Tensor(x_train.data)
            ).to_dense()
        else:
            x_train = torch.Tensor(x_train)
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
                scheduler.step()
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
            if gen_loss > earlystop_a:
                if patienti >= patient:
                    logging.info('Apply earlystop.')
                    break
                else:
                    patienti += 1
        
        pickle.dump(bestmodel, open(f'model/{modelname}.pkl', 'wb'))
        logging.info(f'model saved in model/{modelname}.pkl')
            

        


        


        