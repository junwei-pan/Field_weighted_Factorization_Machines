import time 
import sys
import os
import shutil
import json
from sklearn.metrics import roc_auc_score
from scipy.sparse import coo_matrix
import tensorflow as tf
import numpy as np
from time import gmtime, strftime
import pickle as pkl
from itertools import islice
from conf.conf_fwfm import *
from conf.conf_ffm import *
from conf.conf_lr import *
from conf.conf_fm import *
#from conf.conf_fwfmoh import *

import utils
from models import LR, FM, PNN1, PNN1_Fixed, PNN2, FNN, CCPM, Fast_CTR, Fast_CTR_Concat, FwFM, FFM, FwFM_LE

# Criteo CTR data set
#train_file = '/tmp/jwpan/data_criteo/train.txt.train.thres20.yx'
#test_file = '/tmp/jwpan/data_criteo/train.txt.validation.thres20.yx'
#test_file = '/tmp/jwpan/data_criteo/train.txt.test.thres20.yx'
#train_file = '/tmp/jwpan/data_criteo/train.txt.train.thres20.ffm10.0.yx'
#test_file = '/tmp/jwpan/data_criteo/train.txt.test.thres20.ffm10.0.yx'

# Yahoo CTR data set
train_file = '../data_yahoo/dataset2/ctr_20170517_0530_0.015.txt.thres10.yx.10k'
test_file = '../data_yahoo/dataset2/ctr_20170531.txt.downsample_all.0.1.thres10.yx.10k'
#train_file = '../data_yahoo/dataset2/ctr_20170517_0530_0.015.txt.thres10.yx'
#test_file = '../data_yahoo/dataset2/ctr_20170531.txt.downsample_all.0.1.thres10.yx'
#test_file = '/tmp/jwpan/data_yahoo/dataset2/ctr_20170531.txt.downsample_all.0.1.thres10.yx'
#train_file = '/tmp/jwpan/data_yahoo/dataset2/ctr_20170517_0530_0.015.txt.thres10.ffm2.8.yx'
#test_file = '/tmp/jwpan/data_yahoo/dataset2/ctr_20170601.txt.downsample_all.0.1.thres10.ffm2.8.yx'
#train_file = '/tmp/jwpan/data_yahoo/dataset2/ctr_20170517_0530_0.015.txt.thres10.ffm12.6.yx'
#test_file = '/tmp/jwpan/data_yahoo/dataset2/ctr_20170531.txt.downsample_all.0.1.thres10.ffm12.6.yx'
#test_file = '/tmp/jwpan/data_yahoo/dataset2/ctr_20170601.txt.downsample_all.0.1.thres10.yx'

# fm_model_file = '../data/fm.model.txt'
print "train_file: ", train_file
print "test_file: ", test_file
sys.stdout.flush()

path_feature_map = '../data_yahoo/dataset2/featindex_25m_thres10.txt'
path_saved_model = 'model'
if os.path.exists(path_saved_model) and os.path.isdir(path_saved_model):
    shutil.rmtree(path_saved_model)
INPUT_DIM, FIELD_OFFSETS = utils.initiate(path_feature_map)

train_label = utils.read_label(train_file)
test_label = utils.read_label(test_file)

train_size = train_label.shape[0]
test_size = test_label.shape[0]
num_feas = len(utils.FIELD_SIZES)

min_round = 1
num_round = 1000
early_stop_round = 2
batch_size = 2000
#bb = 10
round_no_improve = 5

field_offsets = utils.FIELD_OFFSETS

def train(model, name):
    builder = tf.saved_model.builder.SavedModelBuilder('model')
    global batch_size, time_run, time_read, time_process
    history_score = []
    start_time = time.time()
    print 'epochs\tloss\ttrain-auc\teval-auc\ttime'
    sys.stdout.flush()
    for i in range(num_round):
        fetches = [model.optimizer, model.loss]
        if batch_size > 0:
            ls = []
            f = open(train_file, 'r')
            lst_lines = []
            for line in f:
                if len(lst_lines) < batch_size:
                    lst_lines.append(line)
                else:
                    X_i, y_i = utils.slice(utils.process_lines(lst_lines, name, INPUT_DIM, FIELD_OFFSETS), 0, -1) # type of X_i, X_i[0], X_i[0][0] is list, tuple and np.ndarray respectively.
                    _, l = model.run(fetches, X_i, y_i)
                    ls.append(l)
                    lst_lines = [line]
            f.close()
            if len(lst_lines) > 0:
                X_i, y_i = utils.slice(utils.process_lines(lst_lines, name, INPUT_DIM, FIELD_OFFSETS), 0, -1)
                _, l = model.run(fetches, X_i, y_i)
                ls.append(l)
        elif batch_size == -1:
            pass
        lst_train_pred = []
        lst_test_pred = []
        if batch_size > 0:
            f = open(train_file, 'r')
            lst_lines = []
            for line in f:
                if len(lst_lines) < batch_size:
                    lst_lines.append(line)
                else:
                    X_i, y_i = utils.slice(utils.process_lines(lst_lines, name, INPUT_DIM, FIELD_OFFSETS), 0, -1)
                    _train_preds = model.run(model.y_prob, X_i)
                    lst_train_pred.append(_train_preds)
                    lst_lines = [line]
            f.close()
            if len(lst_lines) > 0:
                X_i, y_i = utils.slice(utils.process_lines(lst_lines, name, INPUT_DIM, FIELD_OFFSETS), 0, -1)
                _train_preds = model.run(model.y_prob, X_i)
                lst_train_pred.append(_train_preds)
            f = open(test_file, 'r')
            lst_lines = []
            for line in f:
                if len(lst_lines) < batch_size:
                    lst_lines.append(line)
                else:
                    X_i, y_i = utils.slice(utils.process_lines(lst_lines, name, INPUT_DIM, FIELD_OFFSETS), 0, -1)
                    _test_preds = model.run(model.y_prob, X_i)
                    lst_test_pred.append(_test_preds)
                    lst_lines = [line]
            f.close()
            if len(lst_lines) > 0:
                X_i, y_i = utils.slice(utils.process_lines(lst_lines, name, INPUT_DIM, FIELD_OFFSETS), 0, -1)
                _test_preds = model.run(model.y_prob, X_i)
                lst_test_pred.append(_test_preds)
        train_preds = np.concatenate(lst_train_pred)
        test_preds = np.concatenate(lst_test_pred)
        train_score = roc_auc_score(train_label, train_preds)
        test_score = roc_auc_score(test_label, test_preds)
        print '%d\t%f\t%f\t%f\t%f\t%s' % (i, np.mean(ls), train_score, test_score, time.time() - start_time, strftime("%Y-%m-%d %H:%M:%S", gmtime()))
        sys.stdout.flush()
        if i == 0:
            map = {}
            for i in range(15):
                map["field_" + str(i) + "_indices"] = model.X[i].indices
                map["field_" + str(i) + "_values"] = model.X[i].values
                map["field_" + str(i) + "_dense_shape"] = model.X[i].dense_shape
            print "save!"
            print model.y_prob.name
            builder.add_meta_graph_and_variables(
                model.sess, 
                [tf.saved_model.tag_constants.SERVING],
                signature_def_map = {
                    "model": tf.saved_model.signature_def_utils.predict_signature_def(
                        inputs = map,
                        outputs = {"y": model.y_prob})
                    })
            builder.save()
        if i == 12:
            break

def mapConf2Model(name):
    conf = d_name_conf[name]
    model_name = name.split('_')[0]
    if model_name == 'ffm':
        return FFM(**conf)
    elif model_name == 'fwfm':
        return FwFM(**conf)
    elif model_name == 'fm':
        return FM(**conf)
    elif model_name == 'lr':
        return LR(**conf)
    elif model_name == 'fwfmoh':
        return FwFM_LE(**conf)

#for name in ['ffm_l2_v_1e-7_lr_1e-1', 'ffm_l2_v_1e-7_lr_1e-2', 'ffm_l2_v_1e-7_lr_1e-3', 'ffm_l2_v_1e-7_lr_1e-4', 'ffm_l2_v_1e-7_lr_1e-5', 'ffm_l2_v_1e-7_lr_1e-6']:
#for name in ['lr_l2_1e-7', 'lr_l2_1e-8', 'lr_l2_1e-9']:
#for name in ['lr_l2_1e-1', 'lr_l2_1e-2', 'lr_l2_1e-3', 'lr_l2_1e-4', 'lr_l2_1e-5', 'lr_l2_1e-6', 'lr_l2_1e-7', 'lr_l2_1e-8']:
#for name in ['fm_l2_v_1e-1', 'fm_l2_v_1e-2', 'fm_l2_v_1e-3', 'fm_l2_v_1e-4', 'fm_l2_v_1e-5', 'fm_l2_v_1e-6', 'fm_l2_v_1e-7', 'fm_l2_v_1e-8']:
#for name in ['ffm_l2_v_1e-7_lr_1e-4']:
#for name in ['fwfm_l2_v_1e-1', 'fwfm_l2_v_1e-2', 'fwfm_l2_v_1e-3', 'fwfm_l2_v_1e-4', 'fwfm_l2_v_1e-5', 'fwfm_l2_v_1e-6', 'fwfm_l2_v_1e-7', 'fwfm_l2_v_1e-8']:
#for name in ['lr_l2_1e-7_lr_1e-5', 'lr_l2_1e-7_lr_1e-6', 'lr_l2_1e-7_lr_1e-7', 'lr_l2_1e-7_lr_1e-8']:
#for name in ['ffm_l2_v_1e-1', 'ffm_l2_v_1e-2', 'ffm_l2_v_1e-3', 'ffm_l2_v_1e-4', 'ffm_l2_v_1e-5', 'ffm_l2_v_1e-6', 'ffm_l2_v_1e-7', 'ffm_l2_v_1e-8']:
#for name in ['ffm_l2_v_1e-8']:
#for name in ['fwfm_l2_v_1e-5_lr_1e-7']:
#for name in ['fwfm_l2_v_1e-4', 'fwfm_l2_v_1e-5', 'fwfm_l2_v_1e-6', 'fwfm_l2_v_1e-7', 'fwfm_l2_v_1e-8']:
#for name in ['fwfm_l2_v_1e-6']:
#for name in ['fm_l2_v_1e-6']:
for name in ['fwfm_l2_v_1e-5']:
    print 'name with none activation', name
    sys.stdout.flush()
    model = mapConf2Model(name)
    train(model, name + '_yahoo_dataset2.2')
    #train(model, name + '_criteo')
