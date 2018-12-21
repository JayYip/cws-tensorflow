# Tensorflow中文分词模型

注: 如果对准确度比较高的要求, 请使用 https://github.com/JayYip/bert-multitask-learning

部分代码参考 [TensorFlow Model Zoo](https://github.com/tensorflow/models)

运行环境:

- Python 3.5 / Python 2.7
- Tensorflow r1.4
- Windows / Ubuntu 16.04
- hanziconv 0.3.2
- numpy

## 训练模型

### 1. 建立训练数据
进入到data目录下，执行以下命令

```
DATA_OUTPUT="output_dir"

python build_pku_msr_input.py \ 
    --num_threads=4 \
    --output_dir=${DATA_OUTPUT}
```

### 2. 字符嵌入

#### 2.1 预训练好的字嵌入
1. 将`configuration.py`中的`ModelConfig`的`self.random_embedding`设置为`False`
2. 从[Polygot](https://sites.google.com/site/rmyeid/projects/polyglot)下载中文字嵌入数据集至项目目录，运行项目目录下`process_chr_embedding.py`。

```
EMBEDDING_DIR=...
VOCAB_DIR=...

python process_chr_embedding.py \
    --chr_embedding_dir=${EMBEDDING_DIR}
    --vocab_dir=${VOCAB_DIR}
```

#### 2.2 随机初始化字嵌入

将`configuration.py`中的`ModelConfig`的`self.random_embedding`设置为`True`

### 3. 训练模型

根据需要修改configuration.py里面的模型及训练参数，开始训练模型。
以下参数如不提供将会使用默认值。

```
TRAIN_INPUT="data\${DATA_OUTPUT}"
MODEL="save_model"

python train.py \
    --input_file_dir=${TRAIN_INPUT} \
    --train_dir=${MODEL} \
    --log_every_n_steps=10
    
```

## 使用训练好的模型进行分词

编码须为utf8，检测的后缀为'txt'，'csv'， 'utf8'。

```
INF_INPUT=...
INF_OUTPUT=...

python inference.py \
    --input_file_dir=${INF_INPUT} \
    --train_dir=${MODEL} \
    --vocab_dir=${VOCAB_DIR} \
    --out_dir=${INF_OUTPUT}
```

## 如何根据自己需要修改算法

本模型使用的是单向LSTM+CRF，但是提供了算法修改的可能性。在```lstm_based_cws_model.py```文件中的



