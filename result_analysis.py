import os
import re
import pandas as pd

result = {'version':[], 'val_acc':[], 'early_stop':[]}

for logfile in os.listdir('./log'):
    if logfile[:3] == 'mlp':
        version = logfile[4:6]
        with open(os.path.join('log', logfile), 'r', encoding='utf-8') as f:
            content = f.read()
            matches = re.findall(r'accuracy: (\d+\.\d+)', content)
            if matches:
                test_acc = float(matches[-1])
                print(test_acc)
            else: 
                test_acc = -1
            matches = re.findall(r'(={19}第\d+轮开始={23})', content)
            if matches:
                early_stop = matches[-1][20:22]
            else:
                early_stop = -1

        result['version'].append(version)
        result['val_acc'].append(test_acc)
        result['early_stop'].append(early_stop)

pd.DataFrame(result).to_csv('result.csv', index=False)
  
            