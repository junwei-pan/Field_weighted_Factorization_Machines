import time 
import sys
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

import utils
from models import LR, FM, PNN1, PNN1_Fixed, PNN2, FNN, CCPM, Fast_CTR, Fast_CTR_Concat, FwFM, FFM

# Criteo CTR data set
'''
train_file = '/tmp/jwpan/data_criteo/train.txt.train.thres20.yx.100k'
test_file = '/tmp/jwpan/data_criteo/train.txt.validation.thres20.yx.100k'
'''

# Yahoo CTR data set
train_file = '/tmp/jwpan/data_yahoo/dataset2/ctr_20170517_0530_0.015.txt.thres10.yx.100k'
test_file = '/tmp/jwpan/data_yahoo/dataset2/ctr_20170531.txt.downsample_all.0.1.thres10.yx.100k'
'''
#train_file = '/tmp/jwpan/data_yahoo/dataset2/ctr_20170517_0530_0.015.txt.thres10.ffm12.6.yx'
#test_file = '/tmp/jwpan/data_yahoo/dataset2/ctr_20170531.txt.downsample_all.0.1.thres10.ffm12.6.yx'
#test_file = '/tmp/jwpan/data_yahoo/dataset2/ctr_20170601.txt.downsample_all.0.1.thres10.yx'
'''

# fm_model_file = '../data/fm.model.txt'
print "train_file: ", train_file
print "test_file: ", test_file
sys.stdout.flush()

input_dim = utils.INPUT_DIM

train_data = utils.read_data(train_file)
test_data = utils.read_data(test_file)
train_data = utils.split_data(train_data)
test_data = utils.split_data(test_data)

train_label = utils.read_label(train_file)
test_label = utils.read_label(test_file)

train_size = train_label.shape[0]
test_size = test_label.shape[0]
num_feas = len(utils.FIELD_SIZES)

min_round = 1
num_round = 1000
early_stop_round = 2
batch_size = 2000
bb = 10
round_no_improve = 5

field_offsets = utils.FIELD_OFFSETS

time_run = 0
time_process = 0

def train(model, name):
    global batch_size
    global time_run
    global time_process
    history_score = []
    start_time = time.time()
    print 'epochs\tloss\ttrain-auc\teval-auc\ttime'
    sys.stdout.flush()
    for i in range(num_round):
        fetches = [model.optimizer, model.loss]
        if batch_size > 0:
            ls = []
            for j in range(train_size / batch_size + 1):
                start_process = time.time()
                X_i, y_i = utils.slice(train_data, j * batch_size, batch_size)
                time_process += time.time() - start_process
                start_run = time.time()
                _, l = model.run(fetches, X_i, y_i)
                time_run += time.time() - start_run
                ls.append(l)
            '''
            f = open(train_file, 'r')
            lst_lines = []
            for line in f:
                if len(lst_lines) < batch_size:
                    lst_lines.append(line)
                else:
                    X_i, y_i = utils.slice(utils.process_lines(lst_lines, name), 0, -1)
                    print 'type(X_i)', type(X_i)
                    print 'type(X_i[0])', type(X_i[0])
                    _, l = model.run(fetches, X_i, y_i)
                    ls.append(l)
                    lst_lines = [line]
            f.close()
            if len(lst_lines) > 0:
                X_i, y_i = utils.slice(utils.process_lines(lst_lines, name), 0, -1)
                _, l = model.run(fetches, X_i, y_i)
                ls.append(l)
            '''
            '''
            while True:
                lines_gen = list(islice(f, batch_size * bb))
                if not lines_gen:
                    break
                for ib in range(bb):
                    X_i, y_i = utils.slice(utils.process_lines(lines_gen[batch_size * ib : batch_size * (ib+1)], name), 0, -1)
                    _, l = model.run(fetches, X_i, y_i)
                    ls.append(l)
            '''
        elif batch_size == -1:
            pass
            """
            X_i, y_i = utils.slice(train_data)
            _, l = model.run(fetches, X_i, y_i)
            ls = [l]
            """
        lst_train_pred = []
        lst_test_pred = []
        if batch_size > 0:
            f = open(train_file, 'r')
            lst_lines = []
            for line in f:
                if len(lst_lines) < batch_size:
                    lst_lines.append(line)
                else:
                    X_i, y_i = utils.slice(utils.process_lines(lst_lines, name), 0, -1)
                    _train_preds = model.run(model.y_prob, X_i)
                    lst_train_pred.append(_train_preds)
                    lst_lines = [line]
            f.close()
            if len(lst_lines) > 0:
                X_i, y_i = utils.slice(utils.process_lines(lst_lines, name), 0, -1)
                _train_preds = model.run(model.y_prob, X_i)
                lst_train_pred.append(_train_preds)
            '''
            while True:
                lines_gen = list(islice(f, batch_size * bb))
                if not lines_gen:
                    break
                for ib in range(bb):
                    X_i, y_i = utils.slice(utils.process_lines(lines_gen[batch_size * ib : batch_size * (ib+1)], name), 0, -1)
                    _train_preds = model.run(model.y_prob, X_i)
                    lst_train_pred.append(_train_preds)
            '''
            """
            for j in range(train_size / batch_size + 1):
                X_i, y_i = utils.slice(train_data, j * batch_size, batch_size)
                #X_i = utils.libsvm_2_coo(X_i, (len(X_i), input_dim)).tocsr()
                _train_preds = model.run(model.y_prob, X_i)
                lst_train_pred.append(_train_preds)
            """
            f = open(test_file, 'r')
            lst_lines = []
            for line in f:
                if len(lst_lines) < batch_size:
                    lst_lines.append(line)
                else:
                    X_i, y_i = utils.slice(utils.process_lines(lst_lines, name), 0, -1)
                    _test_preds = model.run(model.y_prob, X_i)
                    lst_test_pred.append(_test_preds)
                    lst_lines = [line]
            f.close()
            if len(lst_lines) > 0:
                X_i, y_i = utils.slice(utils.process_lines(lst_lines, name), 0, -1)
                _test_preds = model.run(model.y_prob, X_i)
                lst_test_pred.append(_test_preds)
            '''
            while True:
                lines_gen = list(islice(f, batch_size * bb))
                if not lines_gen:
                    break
                for ib in range(bb):
                    X_i, y_i = utils.slice(utils.process_lines(lines_gen[batch_size * ib : batch_size * (ib+1)], name), 0, -1)
                    _test_preds = model.run(model.y_prob, X_i)
                    lst_test_pred.append(_test_preds)
            '''
            """
            for j in range(test_size / batch_size + 1):
                X_i, y_i = utils.slice(test_data, j * batch_size, batch_size)
                #X_i = utils.libsvm_2_coo(X_i, (len(X_i), input_dim)).tocsr()
                _test_preds = model.run(model.y_prob, X_i)
                lst_test_pred.append(_test_preds)
            """
        train_preds = np.concatenate(lst_train_pred)
        test_preds = np.concatenate(lst_test_pred)
        train_score = roc_auc_score(train_label, train_preds)
        test_score = roc_auc_score(test_label, test_preds)
        print '%d\t%f\t%f\t%f\t%f\t%s' % (i, np.mean(ls), train_score, test_score, time.time() - start_time, strftime("%Y-%m-%d %H:%M:%S", gmtime()))
        print 'time_run', time_run
        print 'time_process', time_process
        # Save the model to local files
        '''
        path_model = 'model/' + str(name) + '_epoch_' + str(i)
        model.dump(path_model)
        d_label_score = {}
        d_label_score['label'] = test_label
        d_label_score['score'] = test_preds
        #path_label_score = 'model/label_score_' + str(name) + '_epoch_' + str(i)
        #pkl.dump(d_label_score, open(path_label_score, 'wb'))
        sys.stdout.flush()
        '''
        history_score.append(test_score)
        if i > min_round and i > early_stop_round:
            i_max = np.argmax(history_score)
            # Early stop
            if i - i_max >= early_stop_round:
                print 'early stop\nbest iteration:\n[%d]\teval-auc: %f' % (
                    np.argmax(history_score), np.max(history_score))
                sys.stdout.flush()
                break
            # No improvement for round_no_improve rounds
            elif history_score[-1] - history_score[max(-1 * round_no_improve, -1 * len(history_score))] < 1e-4:
                print 'no improvement for %d rounds. \nbest iteration:\n[%d]\teval-auc: %f' % (
                    round_no_improve, np.argmax(history_score), np.max(history_score))
                sys.stdout.flush()
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

#for name in ['fmnn_3way', 'pnn1_fixed_0.0005_no_field_bias', 'pnn1_fixed_0.0005_dropout']:
#for name in ['fwfm_0.0005', 'fwfm_0.0005_without_field_bias']:
#for name in ['pnn1_fixed_0.0005_no_field_bias', 'fwfm_0.0001_without_field_bias', 'fwfm_0.0005_without_field_bias_20']:
#for name in ['pnn1', 'pnn2', 'fwfm', 'fwfm_with_field_bias', 'fwfm_0.001_l2_v', 'fwfm_0.001_l2_r', 'fwfm_0.001_l2_vr']:
#for name in ['fwfm_l2_0.00005_v_lr_0.002', 'fwfm_l2_0.00002_v_lr_0.002', 'fwfm_l2_0.00001_v_lr_0.002']:
#for name in ['fwfm_l2_0.000005_v_lr_0.002', 'fwfm_l2_0.000001_v_lr_0.002']:
#for name in ['fwfm_l2_0.0000005_v_lr_0.002', 'fwfm_l2_0.0000001_v_lr_0.002', 'fwfm_l2_0.00000001_v_lr_0.002']:
#for name in ['fwfm_k_5', 'fwfm_k_15', 'fwfm_k_20', 'fwfm_k_30', 'fwfm_k_50', 'fwfm_k_100', 'fwfm_k_200']:
#for name in ['fwfm_k_5_lr_0.0001', 'fwfm_k_15_lr_0.0001', 'fwfm_k_20_lr_0.0001', 'fwfm_k_30_lr_0.0001', 'fwfm_k_50_lr_0.0001', 'fwfm_k_100_lr_0.0001', 'fwfm_k_200_lr_0.0001']:
#for name in ['fwfm_r_l2_0.005', 'fwfm_r_l2_0.001', 'fwfm_r_l2_0.0005', 'fwfm_r_l2_0.0001', 'fwfm_r_l2_0.00005', 'fwfm_r_l2_0.00001']:
#for name in ['fwfm_k_10_lr_0.0001']:
#for name in ['lr', 'fm']:
#for name in ['fwfm_gd', 'fwfm_momentum', 'fwfm_nestorov', 'fwfm_adagrad', 'fwfm_adadelta']:
#for name in ['fwfm_lr_0.05', 'fwfm_lr_0.01', 'fwfm_lr_0.005', 'fwfm_lr_0.001', 'fwfm_lr_0.0005', 'fwfm_lr_0.0001', 'fwfm_lr_0.00005', 'fwfm_lr_0.00001', 'fwfm_lr_0.000005', 'fwfm_lr_0.000001']:
#for name in ['fwfm_r_l2_0.1', 'fwfm_r_l2_0.01', 'fwfm_r_l2_0.001', 'fwfm_r_l2_0.0001', 'fwfm_r_l2_0.00001', 'fwfm_r_l2_0.000001', 'fwfm_r_l2_0.0000001', 'fwfm_v_l2_0.1', 'fwfm_v_l2_0.01', 'fwfm_v_l2_0.001', 'fwfm_v_l2_0.0001', 'fwfm_v_l2_0.00001', 'fwfm_v_l2_0.000001', 'fwfm_v_l2_0.0000001']:
#for name in ['fwfm_v_l2_0.00001', 'fwfm_v_l2_0.000001', 'fwfm_v_l2_0.0000001']:
#for name in ['fwfm']:
#for name in ['fwfm_v_l2_0.1', 'fwfm_v_l2_0.01', 'fwfm_v_l2_0.001', 'fwfm_v_l2_0.0001', 'fwfm_v_l2_0.00001', 'fwfm_v_l2_0.000001', 'fwfm_v_l2_0.0000001']:
#for name in ['fwfm_w_l2_0.1', 'fwfm_w_l2_0.01', 'fwfm_w_l2_0.001', 'fwfm_w_l2_0.0001', 'fwfm_w_l2_0.00001', 'fwfm_w_l2_0.000001']:
#for name in ['fwfm_adam']:
#for name in ['fm_0.01', 'fm_0.001', 'fm_0.0001', 'fm_0.0001']:
#for name in ['ffm_l2_v_0.000001', 'ffm_l2_v_0.0000001', 'ffm_l2_v_0.00000001']:
#for name in ['ffm_l2_v_0.00001']:
#for name in ['ffm_l2_v_1e-7_lr_1e-1', 'ffm_l2_v_1e-7_lr_1e-2', 'ffm_l2_v_1e-7_lr_1e-3', 'ffm_l2_v_1e-7_lr_1e-4', 'ffm_l2_v_1e-7_lr_1e-5', 'ffm_l2_v_1e-7_lr_1e-6']:
#for name in ['fwfm_l2_v_1e-1', 'fwfm_l2_v_1e-2', 'fwfm_l2_v_1e-3', 'fwfm_l2_v_1e-4', 'fwfm_l2_v_1e-5', 'fwfm_l2_v_1e-6', 'fwfm_l2_v_1e-7', 'fwfm_l2_v_1e-8']:
#for name in ['lr_l2_1e-7', 'lr_l2_1e-8', 'lr_l2_1e-9']:
#for name in ['fwfm_l2_v_1e-1', 'fwfm_l2_v_1e-2', 'fwfm_l2_v_1e-3', 'fwfm_l2_v_1e-4', 'fwfm_l2_v_1e-5', 'fwfm_l2_v_1e-6', 'fwfm_l2_v_1e-7', 'fwfm_l2_v_1e-8']:
#for name in ['ffm_l2_v_1e-1', 'ffm_l2_v_1e-2', 'ffm_l2_v_1e-3', 'ffm_l2_v_1e-4', 'ffm_l2_v_1e-5', 'ffm_l2_v_1e-6', 'ffm_l2_v_1e-7', 'ffm_l2_v_1e-8']:
#for name in ['lr_l2_1e-1', 'lr_l2_1e-2', 'lr_l2_1e-3', 'lr_l2_1e-4', 'lr_l2_1e-5', 'lr_l2_1e-6', 'lr_l2_1e-7', 'lr_l2_1e-8']:
#for name in ['fm_l2_v_1e-1', 'fm_l2_v_1e-2', 'fm_l2_v_1e-3', 'fm_l2_v_1e-4', 'fm_l2_v_1e-5', 'fm_l2_v_1e-6', 'fm_l2_v_1e-7', 'fm_l2_v_1e-8']:
for name in ['ffm_l2_v_1e-7_lr_1e-4']:
    print 'name with none activation', name
    sys.stdout.flush()
    model = mapConf2Model(name)
    #train(model, name + '_yahoo_dataset2.2')
    train(model, name + '_criteo')
