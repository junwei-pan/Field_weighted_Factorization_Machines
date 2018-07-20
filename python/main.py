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
from conf.conf_MTLfwfm import *
from conf.conf_ffm import *
from conf.conf_lr import *
from conf.conf_fm import *
#from conf.conf_fwfmoh import *

import utils
from models import LR, FM, PNN1, PNN1_Fixed, PNN2, FNN, CCPM, Fast_CTR, Fast_CTR_Concat, FwFM, FFM, FwFM_LE, MultiTask_FwFM

# Criteo CTR data set
#train_file = '/tmp/jwpan/data_criteo/train.txt.train.thres20.yx'
#test_file = '/tmp/jwpan/data_criteo/train.txt.validation.thres20.yx'
#test_file = '/tmp/jwpan/data_criteo/train.txt.test.thres20.yx'
#train_file = '/tmp/jwpan/data_criteo/train.txt.train.thres20.ffm10.0.yx'
#test_file = '/tmp/jwpan/data_criteo/train.txt.test.thres20.ffm10.0.yx'

# Yahoo CTR data set
#train_file = '../data_yahoo/dataset2/ctr_20170517_0530_0.015.txt.thres10.yx.10k'
#test_file = '../data_yahoo/dataset2/ctr_20170531.txt.downsample_all.0.1.thres10.yx.10k'
#train_file = '../data_yahoo/dataset2/ctr_20170517_0530_0.015.txt.thres10.yx'
#test_file = '../data_yahoo/dataset2/ctr_20170531.txt.downsample_all.0.1.thres10.yx'
#test_file = '/tmp/jwpan/data_yahoo/dataset2/ctr_20170531.txt.downsample_all.0.1.thres10.yx'
#train_file = '/tmp/jwpan/data_yahoo/dataset2/ctr_20170517_0530_0.015.txt.thres10.ffm2.8.yx'
#test_file = '/tmp/jwpan/data_yahoo/dataset2/ctr_20170601.txt.downsample_all.0.1.thres10.ffm2.8.yx'
#train_file = '/tmp/jwpan/data_yahoo/dataset2/ctr_20170517_0530_0.015.txt.thres10.ffm12.6.yx'
#test_file = '/tmp/jwpan/data_yahoo/dataset2/ctr_20170531.txt.downsample_all.0.1.thres10.ffm12.6.yx'
#test_file = '/tmp/jwpan/data_yahoo/dataset2/ctr_20170601.txt.downsample_all.0.1.thres10.yx'

# Yahoo CVR data set
train_file = '../data_cvr/cvr_imp_20180704_0710_conv_20180704_0716.csv.add_conv_type.thres5.yx.10k'
validation_file = '../data_cvr/cvr_imp_20180711_conv_20180711_0717.csv.add_conv_type.thres5.yx.10k'
test_file = '../data_cvr/cvr_imp_20180712_conv_20180712_0718.csv.add_conv_type.thres5.yx.10k'

# fm_model_file = '../data/fm.model.txt'
print "train_file: ", train_file
print "validation_file: ", validation_file
print "test_file: ", test_file
sys.stdout.flush()

#path_feature_map = '../data_yahoo/dataset2/featindex_25m_thres10.txt'
path_feature_map = '../data_cvr/featureindex_thres5.txt'
#path_saved_model = 'model'
#if os.path.exists(path_saved_model) and os.path.isdir(path_saved_model):
#    shutil.rmtree(path_saved_model)
INPUT_DIM, FIELD_OFFSETS, FIELD_SIZES = utils.initiate(path_feature_map)

print 'FIELD_SIZES', FIELD_SIZES

train_label = utils.read_label(train_file)
validation_label = utils.read_label(validation_file)
test_label = utils.read_label(test_file)

train_size = train_label.shape[0]
validation_size = validation_label.shape[0]
test_size = test_label.shape[0]
num_feas = len(utils.FIELD_SIZES)

min_round = 1
num_round = 1000
early_stop_round = 2
batch_size = 1000
#bb = 10
round_no_improve = 5

field_offsets = utils.FIELD_OFFSETS

def train(model, name, in_memory = True, flag_MTL = True):
    #builder = tf.saved_model.builder.SavedModelBuilder('model')
    global batch_size, time_run, time_read, time_process
    history_score = []
    best_score = -1
    start_time = time.time()
    print 'epochs\tloss\ttrain-auc\teval-auc\ttime'
    sys.stdout.flush()
    if in_memory:
        train_data = utils.read_data(train_file, INPUT_DIM)
        validation_data = utils.read_data(validation_file, INPUT_DIM)
        test_data = utils.read_data(test_file, INPUT_DIM)
        model_name = name.split('_')[0]
        if model_name in ['fnn', 'ccpm', 'pnn1', 'pnn1_fixed', 'pnn2', 'fm', 'fwfm', 'MTLfwfm']:
            train_data = utils.split_data(train_data, FIELD_OFFSETS)
            validation_data = utils.split_data(validation_data, FIELD_OFFSETS)
            test_data = utils.split_data(test_data, FIELD_OFFSETS)
    for i in range(num_round):
        fetches = [model.optimizer, model.loss]
        if batch_size > 0:
            ls = []
            if in_memory:
                for j in range(train_size / batch_size + 1):
                    X_i, y_i = utils.slice(train_data, j * batch_size, batch_size)
                    _, l = model.run(fetches, X_i, y_i)
                    ls.append(l)
            else:
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
        model.dump('model/' + name + '_epoch_' + str(i))
        if in_memory:
            lst_train_preds = []
            lst_validation_preds = []
            lst_test_preds = []
            for j in range(train_size / batch_size + 1):
                X_i, y_i = utils.slice(train_data, j * batch_size, batch_size)
                p = model.run(model.y_prob, X_i, y_i)
                lst_train_preds.append(p)
            for j in range(validation_size / batch_size + 1):
                X_i, y_i = utils.slice(validation_data, j * batch_size, batch_size)
                p = model.run(model.y_prob, X_i, y_i)
                lst_validation_preds.append(p)
            for j in range(test_size / batch_size + 1):
                X_i, y_i = utils.slice(test_data, j * batch_size, batch_size)
                p = model.run(model.y_prob, X_i, y_i)
                lst_test_preds.append(p)
            train_preds = np.concatenate(lst_train_preds)
            validation_preds = np.concatenate(lst_validation_preds)
            test_preds = np.concatenate(lst_test_preds)
            #train_preds = model.run(model.y_prob, utils.slice(train_data)[0])
            #test_preds = model.run(model.y_prob, utils.slice(test_data)[0])
            train_score = roc_auc_score(train_data[1], train_preds)
            validation_score = roc_auc_score(validation_data[1], validation_preds)
            test_score = roc_auc_score(test_data[1], test_preds)
            #print '[%d]\tloss:%f\ttrain-auc: %f\teval-auc: %f' % (i, np.mean(ls), train_score, test_score)
            print '%d\t%f\t%f\t%f\t%f\t%f\t%s' % (i, np.mean(ls), train_score, validation_score, test_score, time.time() - start_time, strftime("%Y-%m-%d %H:%M:%S", gmtime()))
            if flag_MTL:
                d_index_task_label_pred_train = {}
                d_index_task_label_pred_validation = {}
                d_index_task_label_pred_test = {}
                index_task_train = train_data[0][-1].indices
                index_task_validation = validation_data[0][-1].indices
                index_task_test = test_data[0][-1].indices
                for index_tmp in range(len(index_task_train)):
                    index_task = index_task_train[index_tmp]
                    d_index_task_label_pred_train.setdefault(index_task, [[],[]])
                    d_index_task_label_pred_train[index_task][0].append(train_data[1][index_tmp])
                    d_index_task_label_pred_train[index_task][1].append(train_preds[index_tmp])
                for index_task in sorted(list(set(index_task_train))):
                    auc = roc_auc_score(d_index_task_label_pred_train[index_task][0], d_index_task_label_pred_train[index_task][1])
                    print 'train, index_type: %d, number of samples: %d, AUC: %f' % (index_task, len(d_index_task_label_pred_train[index_task][0]), auc)
                for index_tmp in range(len(index_task_validation)):
                    index_task = index_task_validation[index_tmp]
                    d_index_task_label_pred_validation.setdefault(index_task, [[],[]])
                    d_index_task_label_pred_validation[index_task][0].append(validation_data[1][index_tmp])
                    d_index_task_label_pred_validation[index_task][1].append(validation_preds[index_tmp])
                for index_task in sorted(list(set(index_task_validation))):
                    auc = roc_auc_score(d_index_task_label_pred_validation[index_task][0], d_index_task_label_pred_validation[index_task][1])
                    print 'validation, index_type: %d, number of samples: %d, AUC: %f' % (index_task, len(d_index_task_label_pred_validation[index_task][0]), auc)
                for index_tmp in range(len(index_task_test)):
                    index_task = index_task_test[index_tmp]
                    d_index_task_label_pred_test.setdefault(index_task, [[],[]])
                    d_index_task_label_pred_test[index_task][0].append(test_data[1][index_tmp])
                    d_index_task_label_pred_test[index_task][1].append(test_preds[index_tmp])
                for index_task in sorted(list(set(index_task_test))):
                    auc = roc_auc_score(d_index_task_label_pred_test[index_task][0], d_index_task_label_pred_test[index_task][1])
                    print 'test, index_type: %d, number of samples: %d, AUC: %f' % (index_task, len(d_index_task_label_pred_test[index_task][0]), auc)
            history_score.append(validation_score)
            if validation_score < best_score:
                break
            best_score = validation_score
            sys.stdout.flush()
        else:
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
            print 'np.shape(train_preds)', np.shape(train_preds)
            train_score = roc_auc_score(train_label, train_preds)
            test_score = roc_auc_score(test_label, test_preds)
            print '%d\t%f\t%f\t%f\t%f\t%s' % (i, np.mean(ls), train_score, test_score, time.time() - start_time, strftime("%Y-%m-%d %H:%M:%S", gmtime()))
            sys.stdout.flush()
        '''
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
        '''

def mapConf2Model(name):
    conf = d_name_conf[name]
    # 'layer_sizes': [field_sizes, 10, 1],
    conf['layer_sizes'] = [FIELD_SIZES, 10, 1]
    print 'conf', conf
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
    elif model_name == 'MTLfwfm':
        conf['index_lines'] = utils.index_lines
        conf['num_lines'] = FIELD_SIZES[utils.index_lines]
        return MultiTask_FwFM(**conf)

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
#for name in ['fwfm_l2_v_1e-5']:
#for name in ['fwfm_l2_v_1e-5']:
for name in ['MTLfwfm_l2_v_1e-5', 'MTLfwfm_lr_1e-5_l2_v_1e-5', 'MTLfwfm_lr_5e-5_l2_v_1e-5']:
    print 'name with none activation', name
    sys.stdout.flush()
    model = mapConf2Model(name)
    train(model, name + '_yahoo_dataset2.2', True)
    #train(model, name + '_criteo')
