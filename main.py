import argparse
import time
localtime = time.strftime("%Y-%m-%d %H-%M-%S", time.localtime()) 

parser = argparse.ArgumentParser(description='control the experiment.')
parser.add_argument('--train', type=int, default=0, choices=[0,1], help='train or test the model')
parser.add_argument('--version', type=str, default=str(localtime))
parser.add_argument('--model', type=str, choices=['svm', 'rf', 'mlp', 'logisticReg'], default='rf')
parser.add_argument('--vec', type=str, choices=['tf-idf', 'cbow', 'bert'], default='tf-idf')
args = parser.parse_args()

from models import run_model

run_model(args)

