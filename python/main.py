import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.sparse import coo_matrix
import time 
import sys
import tensorflow as tf
from time import gmtime, strftime
import pickle as pkl
from itertools import islice
from conf.conf_fwfm import *

import utils
from models import LR, FM, PNN1, PNN1_Fixed, PNN2, FNN, CCPM, Fast_CTR, Fast_CTR_Concat, FwFM, FFM#, FwFM_LE

#train_file = '../data_cretio/train.txt.thres20.yx.0.7'
#test_file = '../data_cretio/train.txt.thres20.yx.0.3'
#train_file = '../data_cretio/train.txt.100000.yx.0.7'
#test_file = '../data_cretio/train.txt.100000.yx.0.3'
train_file = '/tmp/jwpan/data_yahoo/dataset2/ctr_20170517_0530_0.015.txt.thres10.yx'
test_file = '/tmp/jwpan/data_yahoo/dataset2/ctr_20170531.txt.downsample_all.0.1.thres10.yx'
#train_file = '/tmp/jwpan/data_yahoo/dataset2/ctr_20170517_0530_0.015.txt.thres10.ffm12.6.yx'
#test_file = '/tmp/jwpan/data_yahoo/dataset2/ctr_20170531.txt.downsample_all.0.1.thres10.ffm12.6.yx'
#test_file = '/tmp/jwpan/data_yahoo/dataset2/ctr_20170601.txt.downsample_all.0.1.thres10.yx.1000'
#train_file = '../data_yahoo/ctr_20170517_0530_0.015.txt.thres10.yx.100000'
#test_file = '../data_yahoo/ctr_20170531.txt.downsample_all.0.1.thres10.yx.100000'

# fm_model_file = '../data/fm.model.txt'
print "train_file: ", train_file
print "test_file: ", test_file
sys.stdout.flush()

input_dim = utils.INPUT_DIM

"""
with tf.device('/gpu:0'):
    #tensorflow_dataset = tf.constant(numpy_dataset)
    train_data = tf.constant(utils.read_data(train_file))
    test_data = tf.constant(utils.read_data(test_file))
"""

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

field_offsets = utils.FIELD_OFFSETS

def train(model, name):
    history_score = []
    start_time = time.time()
    print 'epochs\tloss\ttrain-auc\teval-auc\ttime'
    sys.stdout.flush()
    for i in range(num_round):
        fetches = [model.optimizer, model.loss]
        if batch_size > 0:
            ls = []
            f = open(train_file, 'r')
            while True:
                lines_gen = list(islice(f, batch_size * bb))
                if not lines_gen:
                    break
                for ib in range(bb):
                    X_i, y_i = utils.slice(utils.process_lines(lines_gen[batch_size * ib : batch_size * (ib+1)]), 0, -1)
                    _, l = model.run(fetches, X_i, y_i)
                    ls.append(l)
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
            while True:
                lines_gen = list(islice(f, batch_size * bb))
                if not lines_gen:
                    break
                for ib in range(bb):
                    X_i, y_i = utils.slice(utils.process_lines(lines_gen[batch_size * ib : batch_size * (ib+1)]), 0, -1)
                    _train_preds = model.run(model.y_prob, X_i)
                    lst_train_pred.append(_train_preds)
            """
            for j in range(train_size / batch_size + 1):
                X_i, y_i = utils.slice(train_data, j * batch_size, batch_size)
                #X_i = utils.libsvm_2_coo(X_i, (len(X_i), input_dim)).tocsr()
                _train_preds = model.run(model.y_prob, X_i)
                lst_train_pred.append(_train_preds)
            """
            f = open(test_file, 'r')
            while True:
                lines_gen = list(islice(f, batch_size * bb))
                if not lines_gen:
                    break
                for ib in range(bb):
                    X_i, y_i = utils.slice(utils.process_lines(lines_gen[batch_size * ib : batch_size * (ib+1)]), 0, -1)
                    _test_preds = model.run(model.y_prob, X_i)
                    lst_test_pred.append(_test_preds)
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
        path_model = 'model/' + str(name) + '_epoch_' + str(i)
        path_label_score = 'model/label_score_' + str(name) + '_epoch_' + str(i)
        #model.dump(path_model)
        d_label_score = {}
        d_label_score['label'] = test_label
        d_label_score['score'] = test_preds
        #pkl.dump(d_label_score, open(path_label_score, 'wb'))
        sys.stdout.flush()
        history_score.append(test_score)
        if i > min_round and i > early_stop_round:
            #if np.argmax(history_score) == i - early_stop_round and history_score[-1] - history_score[
            #            -1 * early_stop_round] < 1e-5:
            i_max = np.argmax(history_score)
            if i - i_max >= early_stop_round:
                print 'early stop\nbest iteration:\n[%d]\teval-auc: %f' % (
                    np.argmax(history_score), np.max(history_score))
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
for name in ['fwfm_l2_v_0.1', 'fwfm_l2_v_1e-2', 'fwfm_l2_v_1e-3', 'fwfm_l2_v_1e-4', 'fwfm_l2_v_1e-5', 'fwfm_l2_v_1e-6']:
    print 'name with none activation', name
    sys.stdout.flush()
    model = mapConf2Model(name)
    train(model, 'yahoo_dataset2.2_' + name)
