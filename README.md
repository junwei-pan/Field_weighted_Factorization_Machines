# Cretio Kaggle Display Advertising Challenge Dataset

`wget https://s3-eu-west-1.amazonaws.com/criteo-labs/dac.tar.gz`

## Transfer data format

`python transfer.py`

This script will generate an output data file named ${path_train}.yx and a feature index file named path_fea_index

## Split the data to train/test

`python split_train.py`

This script will split the data to train/test dataset respectively.

## Dataset Stats

  - # positive samples: 32,086,035
  - # negative samples: 13,754,582

Uniq Features for each field:

[551, 92010, 77775, 302, 16, 11594, 624, 3, 32199, 5002, 91955, 3162, 26, 10119, 90453, 10, 4287, 1924, 4, 91489, 16, 15, 39011, 74, 30895, 1436]

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

# Yahoo CTR Dataset

## One Week Dataset(20170524-20170530 for train, 20170531 for test)

Training dataset: 3,571,405 samples, 1,556,404 positive and 2,015,001 negative, positive sample has done 0.003 downsampleing
Testing dataset: 7,990,874 samples, 15,759 positive and 7,975,115 negative, total dataset has done 0.1 downsampling.

## Field and Features:

15 Fields.

Number of uniq features for each field, only those features occurs more than 10 times in the training data will be preserved.

|Fields|#Uniq Features|
|--|--|
|PUBLISHER_ID|5,647|
|PAGE_TLD|9,640|
|SUBDOMAIN|22|
|LAYOUT_ID|24|
|HOUR_OF_DAY|7|
|DAY_OF_WEEK|3|
|GENDER|7|
|AD_POSITION_ID|7|
|AGE_BUCKET|554|
|ADVERTISER_ID|6,605|
|AD_ID|4,964|
|CRRATIVE_ID|4|
|DEVICE_TYPE_ID|1,992|
|LINE_ID|4,771|
|USER_ID|41|

## Performance


