### Cretio Kaggle Display Advertising Challenge Dataset

`wget https://s3-eu-west-1.amazonaws.com/criteo-labs/dac.tar.gz`

#### Transfer data format

`python transfer.py`

This script will generate an output data file named ${path_train}.yx and a feature index file named path_fea_index

#### Split the data to train/test

`python split_train.py`

This script will split the data to train/test dataset respectively.

### Model

LR, FM, FNN, CCPM and PNN are all implemented in `models.py`, based on TensorFlow.

You can train any of the models in `main.py`.

For any questions, please report the issues or contact Junwei Pan(pandevirus@gmail.com).

