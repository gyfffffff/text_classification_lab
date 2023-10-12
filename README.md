# text_classification_lab
## 实验环境

- python 3.9
- scikit-learn 0.22.1
- cuda 11.6
- pytorch 1.12.1

更多包依赖见requirement.txt

## 运行代码
运行测试代码：

`python main.py --version [version] --model [model] --train [0/1] --vec [vectorize-method]`

示例：

```shell
# 测试随机森林01版本并输出测试结果：
python main.py --version 01 --model rf --train 0
```
输出保存在 log/rf_01.txt

```shell
# 用cbow抽取文本特征，并训练mlp, 设置版本为02
python main.py --version 02 --model mlp --train 1 --vec cbow
```
训练时输出保存在 log/mlp_02.txt

**具体参数含义可见 main.py**


运行提交版本的代码：
```shell
# 训练提交版本的模型
python bert_clf.py
# 并生成提交结果submit.py
python submit.py
```
log保存在log/bert.txt, 结果保存在result.txt

运行1-gram+mlp版本的代码:
```python
python 1-gram-test.py
```

bert.py, CBOW.py用于向量化, dataloader.py用于处理数据，它们在models.py中被调用，而main调用models.py。


