# Tensorflow中文分词模型

部分代码参考 [TensorFlow Model Zoo](https://github.com/tensorflow/models)

运行环境:

- Python 3.5
- Tensorflow r1.0
- Windows/Linux?
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

从[Polygot](https://sites.google.com/site/rmyeid/projects/polyglot)下载中文字嵌入数据集至项目目录，运行项目目录下process_chr_embedding.py。

```
python process_chr_embedding.py
```

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
    --out_dir=${INF_OUTPUT}
```





