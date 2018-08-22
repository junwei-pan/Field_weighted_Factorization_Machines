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

For any questions, please report the issues or contact Junwei Pan(jwpan@oath.com).

## Performance

### Models Compare

Config:

factor dimension: 10
optimizer: adam
learning rate: 0.0001
layer activation: tanh, none
layer l2 norm: 0

|Model|Train Loss|Train AUC|Test AUC|
|---|---|---|---|
|IPNN|0.474942|0.770333|0.764046|
|OPNN|0.466014|0.783711|0.771107|
|FwFM|0.462722|0.794266|0.775696|



# Yahoo CTR Dataset

## One Week Dataset(3m)

20170524-20170530 for train, 20170531 for test, please refer to [CTR_Data_Generation](https://git.ouroath.com/jwpan/CTR_Data_Generation) for more information.

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

# Conversion Prediction Data Set

## Generate Data Set based on revenue_fact_orc table

Please refer to repo [CTR Data Generation](https://git.ouroath.com/jwpan/CTR_Data_Generation/tree/master/Conversion_Data_Generation) for details.

## Get conversion type for each line

Run [generate_samples_from_impression_log_and_line_conversion_type.py](https://git.ouroath.com/jwpan/CTR_Data_Generation/blob/master/Conversion_Data_Generation/generate_samples_from_impression_log_and_line_conversion_type.py) to get the conversion type for each line.

## Transfer

Transfer data format.
