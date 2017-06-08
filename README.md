# Cretio Kaggle Display Advertising Challenge Dataset

`wget https://s3-eu-west-1.amazonaws.com/criteo-labs/dac.tar.gz`

## Transfer data format

`python transfer.py`

This script will generate an output data file named ${path_train}.yx and a feature index file named path_fea_index

## Split the data to train/test

`python split_train.py`

This script will split the data to train/test dataset respectively.

## Model

LR, FM, FNN, CCPM and PNN are all implemented in `models.py`, based on TensorFlow.

You can train any of the models in `main.py`.

For any questions, please report the issues or contact Junwei Pan(pandevirus@gmail.com).

## Performance

### Models Compare

Config:

factor dimension: 10
optimizer: adam
learning rate: 0.0001
layer activation: tanh, none
layer l2 norm: 0

|Model|AUC or ROC|
|---|---|
|fast_ctr|0.76505|
|fast_ctr_concat|0.765309|
|fnn|0.765309|
|pnn1|0.767347|
|pnn2|0.769773|
|pnn1_fixed|0.775139|

### Fine Tune pnn1_fixed

|Parameter| AUC|
|---|---|
|lr=0.001|0.773991|
|lr=0.001, k=5|0.772792|
|lr=0.001, k=20|0.774981|
|lr=0.001, k=50|0.775141|
|lr=0.001, gd|0.626260|
|lr=0.001, drouput=0.5|0.632780|
|lr=0.001, drouput=0.5, l2=0.5|0.622358|
