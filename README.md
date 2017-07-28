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

|Model|Train Loss|Train AUC|Test AUC|
|---|---|---|---|---|
|IPNN|0.474942|0.770333|0.764046|
|OPNN|0.466014|0.783711|0.771107|
|FwFM|0.462722|0.794266|0.775696|



# Yahoo CTR Dataset

## One Week Dataset(3m)

20170524-20170530 for train, 20170531 for test, please refer to [CTR_Data_Generation](https://git.corp.yahoo.com/jwpan/CTR_Data_Generation) for more information.

Training dataset: 3,571,405 samples, 1,556,404 positive and 2,015,001 negative, positive sample has done 0.003 downsampleing. 34,288 unique features.

Testing dataset: 7,990,874 samples, 15,759 positive and 7,975,115 negative, total dataset has done 0.1 downsampling.

There are totally 15 Fields, and 34,288 unique features(with number of occurence more than 10), following is the number of unique features for each field:

|Fields|#Uniq Features|
|---|---|
|PUBLISHER_ID|41|
|PAGE_TLD|5,647|
|SUBDOMAIN|9,640|
|LAYOUT_ID|22|
|HOUR_OF_DAY|24|
|DAY_OF_WEEK|7|
|GENDER|3|
|AD_POSITION_ID|7|
|AGE_BUCKET|7|
|ADVERTISER_ID|554|
|AD_ID|6,605|
|CRRATIVE_ID|4,964|
|DEVICE_TYPE_ID|4|
|LINE_ID|1,992|
|USER_ID|4,771|

## Performance

|Model|AUC|
|---|---|
|PNN1|0.848153|
|PNN2|0.854713|
|PNN1_Fixed|0.857946|

## Two Weeks Dataset(25m)

20170517-20170530 for train, 20170531 for test.

Training dataset: 24,885,731 samples, 3,283,760 positive and 21,601,971 negative, positive sample has done 0.015 downsampleing

Testing dataset: 7,990,874 samples, 15,759 positive and 7,975,115 negative, total dataset has done 0.1 downsampling.

There are totally 15 Fields, and 156,393 unique features(with number of occurence more than 10), following is the number of unique features for each field:

|Fields|#Uniq Features|
|---|---|
|PUBLISHER_ID|43|
|PAGE_TLD|18,576|
|SUBDOMAIN|43,187|
|LAYOUT_ID|24|
|HOUR_OF_DAY|24|
|DAY_OF_WEEK|7|
|GENDER|3|
|AD_POSITION_ID|8|
|AGE_BUCKET|7|
|ADVERTISER_ID|642|
|AD_ID|10,482|
|CRRATIVE_ID|7,160|
|DEVICE_TYPE_ID|4|
|LINE_ID|2,530|
|USER_ID|73,696|

## Performance

|Model|train loss|train AUC|test AUC|config|
|---|---|---|---|---|
|LR|0.276781|0.858286|0.849761|lr=0.0005,adam|
|VW<sup>[1](#myfootnote1)</sup>|||0.859066|lr=0.0005,adam|
|FM||0.878841|0.861942|lr=0.0005,adam,k=10|
|PNN1||0.861991|0.850883|lr=0.0005,adam,k=10|
|PNN2||0.878399|0.860930|lr=0.0005,adam,k=10|
|FwFM||0.880590|0.864665|lr=0.0005,adam,k=10|
|FwFM w/ field bias||0.881917|0.864626|lr=0.0005,adam,k=10|
|FwFM w/ Dropout|||0.857638|lr=0.0005,adam,k=10|
|FwFM,l2 on v||0.856198|0.852356|lr=0.001,adam,k=10,lambda=0.001|
|FwFM,l2 on r||0.880738|0.864192|lr=0.001,adam,k=10,lambda=0.001|
|FwFM,l2 on v,r||0.839635|0.840396|lr=0.001,adam,k=10,lambda=0.001,0.001|


|k|train loss|Train AUC|Test AUC|
|---|---|---|---|
|5|0.259719|0.879381|0.862137|
|10|0.260862|0.880590|0.864665|
|15|0.258043|0.881974|0.864401|
|20|0.257972|0.882248|0.863971|
|30|0.258461|0.881841|0.864650|
|50|0.257633|0.883046|0.865152|
|100|0.258038|0.883044|0.864571|
|200|0.258909|0.882531|0.864571|

|learning rate|train loss|Train AUC|Test AUC|
|---|---|---|---|---|
|0.05|0.260796|0.880702|0.864731|
|0.01|0.268466|0.874393|0.859382|
|0.005|0.267193|0.877938|0.862593|
|0.001|0.259753|0.882031|0.864831|
|0.0005|0.260796|0.880702|0.864731|
|0.0001|0.259541|0.880036|0.863560|
|0.00005|0.258512|0.880785|0.863121|
|0.00001||||
|0.000005||||
|0.000001||||
|0.0000005||||
|0.0000001||||

<a name="myfootnote1">1</a>: The VW model use addition numerical features(CTR) besides the categorical features. Here is the CTR feature list: TLD_LAYOUT_CTR, TLD_CAMPAIGN_CTR, TLD_IO_CTR, PUB_TLD_LAYOUT_CTR, TLD_CTR, TLD_AD_CTR, PUB_TLD_SEG_AD_CTR
