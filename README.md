# Field-weighted Factorization Machines

## Cretio Kaggle Display Advertising Challenge Dataset

`wget https://s3-eu-west-1.amazonaws.com/criteo-labs/dac.tar.gz`

### Transfer data format

`python transfer.py`

This script will generate an output data file named ${path_train}.yx and a feature index file named path_fea_index

### Split the data to train/test

`python split_train.py`

This script will split the data to train/test dataset respectively.

### Dataset Stats

  - # positive samples: 32,086,035
  - # negative samples: 13,754,582

Uniq Features for each field:

[551, 92010, 77775, 302, 16, 11594, 624, 3, 32199, 5002, 91955, 3162, 26, 10119, 90453, 10, 4287, 1924, 4, 91489, 16, 15, 39011, 74, 30895, 1436]


### Train the model

Generate a conf file like conf/dataset/click_yahoo_dataset2.ini, with the path of feature index file, training, validation and test set, as well as the number of fields.

Run the model by 

```bash
python main.py conf/dataset/click_yahoo_dataset2.ini
```
