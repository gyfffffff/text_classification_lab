from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from mlp import mlp
import torch
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from dataloader import dataloader
import time
import pickle
import logging


def run_model(args):

    version = args.version
    vectorize_method = args.vec
    model = args.model
    train = args.train
    modelname = model+'_'+version

    clf_dict = {'rf': RandomForestClassifier(),
                'svm': svm.SVC(),
                'logisticReg': LogisticRegression()
                }

    tuned_parameters = {
        'rf':[{'n_estimators': [50, 100, 150],
            'max_depth': [12, 24, 30, None],
            'min_samples_leaf': [1, 3, 8, 16, 22], 'min_samples_split': [2, 4, 6, 12, 24]},
          ],
        'svm':[{'C':[1, 10, 100, 1000], 'kernel':['linear']},
            {'C':[1, 10, 100, 1000], 'gamma':[0.0001, 0.001, 0.0], 'kernel':['poly','rbf','sigmoid']},
            ],
        'logisticReg':[
            {'C':[1,10,100,1000], 'multi_class':['multinomial'], 'max_iter':[20000], 'penalty':['l2']},
            {'C':[1,10,100,1000], 'multi_class':['multinomial'], 'max_iter':[20000], 'penalty':['l1'], 'solver':['saga'],
             'C':[1,10,100,1000], 'multi_class':['multinomial'], 'max_iter':[20000], 'penalty':['elasticnet'], 'solver':['saga'], 'l1_ratio':[0.4, 0.6]}
        ]
    }

    logging.basicConfig(format='%(asctime)s  %(message)s',
                filename='log/{}.txt'.format(modelname),
                filemode='a+',
                level=logging.INFO)

    start = time.time()
    if train == True:
        logging.info(f'=================== {modelname} start training ===================')
        X_train, y_train = dataloader(vectorize_method).load_traindata()
        if model != 'mlp':
            # 网格搜索
            logging.info('gridsearching...')
            clf = GridSearchCV(clf_dict[model],
                               tuned_parameters[model],
                               scoring='accuracy',
                               cv=5, n_jobs=3)
            clf.fit(X_train, y_train)
            logging.info('model: {}, best_params: {}'.format(modelname, clf.best_params_))
            means = clf.cv_results_['mean_test_score']
            stds = clf.cv_results_['std_test_score']
            for mean, std, params in zip(means, stds, clf.cv_results_['params']):
                logging.info("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
            
            # 保存模型
            pickle.dump(clf, open('model/{}.pkl'.format(modelname), 'wb'))
            logging.info('model saved in model/{}.pkl'.format(modelname))
        elif model == 'mlp':
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
            logging.info(f"Using {device} device")
            mlp(X_train, y_train, device, modelname, train)
        else:
            logging.info(f'no such model: {modelname}')
    else:
        logging.info(f'=================== {modelname} start testing ===================')
        X_test, y_test = dataloader(vectorize_method).load_testdata()
        try:
            clf = pickle.load(open('model/{}.pkl'.format(modelname), 'rb'))
        except:
            print(
                f'model: {modelname} not found, please check args.version or set args.train 1 first')
            return
        if model != 'mlp':
            predict = clf.predict(X_test)
        elif model == 'mlp':
            device = 'cpu'  
            logging.info(f"Using {device} device")
            
            clf = clf.to(device)
            X_test = X_test.tocoo()
            X_test = torch.sparse_coo_tensor(
                [X_test.row.tolist(), X_test.col.tolist()], torch.Tensor(X_test.data), size=X_test.shape
            ).to_dense().to(device)
            y_test = torch.Tensor(y_test).long().to(device)
            predict = clf(X_test).argmax(1)

        test_acc = accuracy_score(y_test, predict)
        test_f1 = f1_score(y_test, predict, average='macro')
        logging.info('model: {}, test_acc_mean: {}, test_f1_mean: {}'.format(
            modelname, test_acc, test_f1))

    logging.info('time spend {} s'.format(time.time()-start))


