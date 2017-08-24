import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.sparse import coo_matrix
import time 
import sys
import tensorflow as tf
from time import gmtime, strftime
import pickle as pkl

import utils
from models import LR, FM, PNN1, PNN1_Fixed, PNN2, FNN, CCPM, Fast_CTR, Fast_CTR_Concat, FwFM#, FwFM_LE

#train_file = '../data_cretio/train.txt.thres20.yx.0.7'
#test_file = '../data_cretio/train.txt.thres20.yx.0.3'
#train_file = '../data_cretio/train.txt.100000.yx.0.7'
#test_file = '../data_cretio/train.txt.100000.yx.0.3'
#train_file = '../data_yahoo/ctr_20170524_0530_0.003.txt.thres10.yx'
train_file = '/tmp/jwpan/data_yahoo/dataset2/ctr_20170517_0530_0.015.txt.thres10.yx'
test_file = '/tmp/jwpan/data_yahoo/dataset2/ctr_20170531.txt.downsample_all.0.1.thres10.yx'
#train_file = '../data_yahoo/ctr_20170517_0530_0.015.txt.thres10.yx.100000'
#test_file = '../data_yahoo/ctr_20170531.txt.downsample_all.0.1.thres10.yx.100000'

# fm_model_file = '../data/fm.model.txt'
print "train_file: ", train_file
print "test_file: ", test_file
sys.stdout.flush()

input_dim = utils.INPUT_DIM

train_data = utils.read_data(train_file)
# train_data = utils.shuffle(train_data)
test_data = utils.read_data(test_file)

train_size = train_data[0].shape[0]
test_size = test_data[0].shape[0]
num_feas = len(utils.FIELD_SIZES)

min_round = 1
num_round = 1000
early_stop_round = 15
batch_size = 2000

field_sizes = utils.FIELD_SIZES
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
            for j in range(train_size / batch_size + 1):
                X_i, y_i = utils.slice(train_data, j * batch_size, batch_size)
                _, l = model.run(fetches, X_i, y_i)
                ls.append(l)
        elif batch_size == -1:
            X_i, y_i = utils.slice(train_data)
            _, l = model.run(fetches, X_i, y_i)
            ls = [l]
        lst_train_pred = []
        lst_test_pred = []
        if batch_size > 0:
            for j in range(train_size / batch_size + 1):
                X_i, y_i = utils.slice(train_data, j * batch_size, batch_size)
                #X_i = utils.libsvm_2_coo(X_i, (len(X_i), input_dim)).tocsr()
                _train_preds = model.run(model.y_prob, X_i)
                lst_train_pred.append(_train_preds)
            for j in range(test_size / batch_size + 1):
                X_i, y_i = utils.slice(test_data, j * batch_size, batch_size)
                #X_i = utils.libsvm_2_coo(X_i, (len(X_i), input_dim)).tocsr()
                _test_preds = model.run(model.y_prob, X_i)
                lst_test_pred.append(_test_preds)
        train_preds = np.concatenate(lst_train_pred)
        test_preds = np.concatenate(lst_test_pred)
        train_score = roc_auc_score(train_data[1], train_preds)
        test_score = roc_auc_score(test_data[1], test_preds)
        print '%d\t%f\t%f\t%f\t%f\t%s' % (i, np.mean(ls), train_score, test_score, time.time() - start_time, strftime("%Y-%m-%d %H:%M:%S", gmtime()))
        path_model = 'model/' + str(name) + '_epoch_' + str(i)
        path_label_score = 'model/label_score_' + str(name) + '_epoch_' + str(i)
        model.dump(path_model)
        d_label_score = {}
        d_label_score['label'] = test_data[1]
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

train_data = utils.split_data(train_data)
test_data = utils.split_data(test_data)

d_name_model = {}
d_name_model['lr'] = LR(**{
        'input_dim': input_dim,
        'opt_algo': 'adam',
        'learning_rate': 0.0001,
        'l2_weight': 0,
        'random_seed': 0
    })
d_name_model['fm'] = FM(**{
        'input_dim': input_dim,
        'factor_order': 10,
        'opt_algo': 'adam',
        'learning_rate': 0.0001,
        'l2_w': 0,
        'l2_v': 0,
    })
d_name_model['fm_0.005'] = FM(**{
        'input_dim': input_dim,
        'factor_order': 10,
        'opt_algo': 'adam',
        'learning_rate': 0.005,
        'l2_w': 0,
        'l2_v': 0,
    })
'''
d_name_model['fnn'] = FNN(**{
        'layer_sizes': [field_sizes, 10, 1],
        'layer_acts': ['tanh', 'none'],
        'layer_keeps': [1, 1],
        'opt_algo': 'adam',
        'learning_rate': 0.0001,
        'layer_l2': [0, 0],
        'random_seed': 0
    })
'''
d_name_model['pnn1'] = PNN1(**{
        'layer_sizes': [field_sizes, 10, 1],
        'layer_acts': ['tanh', 'none'],
        'layer_keeps': [1, 1],
        'opt_algo': 'adam',
        'learning_rate': 0.0005,
        'layer_l2': [0, 0],
        'kernel_l2': 0,
        'random_seed': 0
    })
'''
d_name_model['pnn1_fixed'] = PNN1_Fixed(**{
        'layer_sizes': [field_sizes, 10, 1],
        'layer_acts': ['tanh', 'none'],
        'layer_keeps': [1, 1],
        'opt_algo': 'adam',
        'learning_rate': 0.0001,
        'layer_l2': [0, 0],
        'kernel_l2': 0,
        'random_seed': 0
    })
d_name_model['pnn1_fixed_0.0005_no_field_bias'] = PNN1_Fixed(**{
        'layer_sizes': [field_sizes, 10, 1],
        'layer_acts': ['tanh', 'none'],
        'layer_keeps': [1, 1],
        'opt_algo': 'adam',
        'learning_rate': 0.0005,
        'layer_l2': [0, 0],
        'kernel_l2': 0,
        'random_seed': 0,
        'has_field_bias': False
    })
'''
d_name_model['pnn2'] = PNN2(**{
        'layer_sizes': [field_sizes, 10, 1],
        'layer_acts': ['tanh', 'none'],
        'layer_keeps': [1, 1],
        'opt_algo': 'adam',
        'learning_rate': 0.0005,
        'layer_l2': [0, 0],
        'kernel_l2': 0,
        'random_seed': 0
    })
'''
d_name_model['fast_ctr_concat'] =  Fast_CTR_Concat(**{
        'layer_sizes': [field_sizes, 10, 1],
        'layer_acts': ['tanh', 'none'],
        'layer_keeps': [1, 1],
        'opt_algo': 'adam',
        'learning_rate': 0.0001,
        'layer_l2': [0, 0],
        'kernel_l2': 0,
        'random_seed': 0
    })
d_name_model['fast_ctr'] = Fast_CTR(**{
        'layer_sizes': [field_sizes, 10, 1],
        'layer_acts': ['tanh', 'none'],
        'layer_keeps': [1, 1],
        'opt_algo': 'adam',
        'learning_rate': 0.0001,
        'layer_l2': [0, 0],
        'kernel_l2': 0,
        'random_seed': 0
})
'''
d_name_model['fwfm'] = FwFM(**{
        'layer_sizes': [field_sizes, 10, 1],
        'layer_acts': ['none', 'none'],
        'layer_keeps': [1, 1],
        'opt_algo': 'adam',
        'learning_rate': 0.0005,
        'layer_l2': [0, 0],
        'kernel_l2': 0,
        'random_seed': 0,
        'has_field_bias': False
    })
d_name_model['fwfm_lr_0.001'] = FwFM(**{
        'layer_sizes': [field_sizes, 10, 1],
        'layer_acts': ['none', 'none'],
        'layer_keeps': [1, 1],
        'opt_algo': 'adam',
        'learning_rate': 0.001,
        'layer_l2': [0, 0],
        'kernel_l2': 0,
        'random_seed': 0,
        'has_field_bias': False
    })
d_name_model['fwfm_lr_0.005'] = FwFM(**{
        'layer_sizes': [field_sizes, 10, 1],
        'layer_acts': ['none', 'none'],
        'layer_keeps': [1, 1],
        'opt_algo': 'adam',
        'learning_rate': 0.005,
        'layer_l2': [0, 0],
        'kernel_l2': 0,
        'random_seed': 0,
        'has_field_bias': False
    })
d_name_model['fwfm_lr_0.01'] = FwFM(**{
        'layer_sizes': [field_sizes, 10, 1],
        'layer_acts': ['none', 'none'],
        'layer_keeps': [1, 1],
        'opt_algo': 'adam',
        'learning_rate': 0.01,
        'layer_l2': [0, 0],
        'kernel_l2': 0,
        'random_seed': 0,
        'has_field_bias': False
    })
d_name_model['fwfm_lr_0.05'] = FwFM(**{
        'layer_sizes': [field_sizes, 10, 1],
        'layer_acts': ['none', 'none'],
        'layer_keeps': [1, 1],
        'opt_algo': 'adam',
        'learning_rate': 0.05,
        'layer_l2': [0, 0],
        'kernel_l2': 0,
        'random_seed': 0,
        'has_field_bias': False
    })
d_name_model['fwfm_lr_0.0005'] = FwFM(**{
        'layer_sizes': [field_sizes, 10, 1],
        'layer_acts': ['none', 'none'],
        'layer_keeps': [1, 1],
        'opt_algo': 'adam',
        'learning_rate': 0.0005,
        'layer_l2': [0, 0],
        'kernel_l2': 0,
        'random_seed': 0,
        'has_field_bias': False
    })
d_name_model['fwfm_lr_0.0001'] = FwFM(**{
        'layer_sizes': [field_sizes, 10, 1],
        'layer_acts': ['none', 'none'],
        'layer_keeps': [1, 1],
        'opt_algo': 'adam',
        'learning_rate': 0.0001,
        'layer_l2': [0, 0],
        'kernel_l2': 0,
        'random_seed': 0,
        'has_field_bias': False
    })
d_name_model['fwfm_lr_0.00005'] = FwFM(**{
        'layer_sizes': [field_sizes, 10, 1],
        'layer_acts': ['none', 'none'],
        'layer_keeps': [1, 1],
        'opt_algo': 'adam',
        'learning_rate': 0.00005,
        'layer_l2': [0, 0],
        'kernel_l2': 0,
        'random_seed': 0,
        'has_field_bias': False
    })
d_name_model['fwfm_lr_0.00001'] = FwFM(**{
        'layer_sizes': [field_sizes, 10, 1],
        'layer_acts': ['none', 'none'],
        'layer_keeps': [1, 1],
        'opt_algo': 'adam',
        'learning_rate': 0.00001,
        'layer_l2': [0, 0],
        'kernel_l2': 0,
        'random_seed': 0,
        'has_field_bias': False
    })
d_name_model['fwfm_lr_0.000005'] = FwFM(**{
        'layer_sizes': [field_sizes, 10, 1],
        'layer_acts': ['none', 'none'],
        'layer_keeps': [1, 1],
        'opt_algo': 'adam',
        'learning_rate': 0.000005,
        'layer_l2': [0, 0],
        'kernel_l2': 0,
        'random_seed': 0,
        'has_field_bias': False
    })
d_name_model['fwfm_lr_0.000001'] = FwFM(**{
        'layer_sizes': [field_sizes, 10, 1],
        'layer_acts': ['none', 'none'],
        'layer_keeps': [1, 1],
        'opt_algo': 'adam',
        'learning_rate': 0.000001,
        'layer_l2': [0, 0],
        'kernel_l2': 0,
        'random_seed': 0,
        'has_field_bias': False
    })
d_name_model['fwfm_k_5'] = FwFM(**{
        'layer_sizes': [field_sizes, 5, 1],
        'layer_acts': ['tanh', 'none'],
        'layer_keeps': [1, 1],
        'opt_algo': 'adam',
        'learning_rate': 0.0005,
        'layer_l2': [0, 0],
        'kernel_l2': 0,
        'random_seed': 0,
        'has_field_bias': False
    })
d_name_model['fwfm_k_15'] = FwFM(**{
        'layer_sizes': [field_sizes, 15, 1],
        'layer_acts': ['tanh', 'none'],
        'layer_keeps': [1, 1],
        'opt_algo': 'adam',
        'learning_rate': 0.0005,
        'layer_l2': [0, 0],
        'kernel_l2': 0,
        'random_seed': 0,
        'has_field_bias': False
    })
d_name_model['fwfm_k_20'] = FwFM(**{
        'layer_sizes': [field_sizes, 20, 1],
        'layer_acts': ['tanh', 'none'],
        'layer_keeps': [1, 1],
        'opt_algo': 'adam',
        'learning_rate': 0.0005,
        'layer_l2': [0, 0],
        'kernel_l2': 0,
        'random_seed': 0,
        'has_field_bias': False
    })
d_name_model['fwfm_k_30'] = FwFM(**{
        'layer_sizes': [field_sizes, 30, 1],
        'layer_acts': ['tanh', 'none'],
        'layer_keeps': [1, 1],
        'opt_algo': 'adam',
        'learning_rate': 0.0005,
        'layer_l2': [0, 0],
        'kernel_l2': 0,
        'random_seed': 0,
        'has_field_bias': False
    })
d_name_model['fwfm_k_50'] = FwFM(**{
        'layer_sizes': [field_sizes, 50, 1],
        'layer_acts': ['tanh', 'none'],
        'layer_keeps': [1, 1],
        'opt_algo': 'adam',
        'learning_rate': 0.0005,
        'layer_l2': [0, 0],
        'kernel_l2': 0,
        'random_seed': 0,
        'has_field_bias': False
    })
d_name_model['fwfm_k_100'] = FwFM(**{
        'layer_sizes': [field_sizes, 100, 1],
        'layer_acts': ['tanh', 'none'],
        'layer_keeps': [1, 1],
        'opt_algo': 'adam',
        'learning_rate': 0.0005,
        'layer_l2': [0, 0],
        'kernel_l2': 0,
        'random_seed': 0,
        'has_field_bias': False
    })
d_name_model['fwfm_k_200'] = FwFM(**{
        'layer_sizes': [field_sizes, 200, 1],
        'layer_acts': ['tanh', 'none'],
        'layer_keeps': [1, 1],
        'opt_algo': 'adam',
        'learning_rate': 0.0005,
        'layer_l2': [0, 0],
        'kernel_l2': 0,
        'random_seed': 0,
        'has_field_bias': False
    })
d_name_model['fwfm_k_5_lr_0.0001'] = FwFM(**{
        'layer_sizes': [field_sizes, 5, 1],
        'layer_acts': ['none', 'none'],
        'layer_keeps': [1, 1],
        'opt_algo': 'adam',
        'learning_rate': 0.0001,
        'layer_l2': [0, 0],
        'kernel_l2': 0,
        'random_seed': 0,
        'has_field_bias': False
    })
d_name_model['fwfm_k_10_lr_0.0001'] = FwFM(**{
        'layer_sizes': [field_sizes, 10, 1],
        'layer_acts': ['none', 'none'],
        'layer_keeps': [1, 1],
        'opt_algo': 'adam',
        'learning_rate': 0.0001,
        'layer_l2': [0, 0],
        'kernel_l2': 0,
        'random_seed': 0,
        'has_field_bias': False
    })
d_name_model['fwfm_k_15_lr_0.0001'] = FwFM(**{
        'layer_sizes': [field_sizes, 15, 1],
        'layer_acts': ['none', 'none'],
        'layer_keeps': [1, 1],
        'opt_algo': 'adam',
        'learning_rate': 0.0001,
        'layer_l2': [0, 0],
        'kernel_l2': 0,
        'random_seed': 0,
        'has_field_bias': False
    })
d_name_model['fwfm_k_20_lr_0.0001'] = FwFM(**{
        'layer_sizes': [field_sizes, 20, 1],
        'layer_acts': ['none', 'none'],
        'layer_keeps': [1, 1],
        'opt_algo': 'adam',
        'learning_rate': 0.0001,
        'layer_l2': [0, 0],
        'kernel_l2': 0,
        'random_seed': 0,
        'has_field_bias': False
    })
d_name_model['fwfm_k_30_lr_0.0001'] = FwFM(**{
        'layer_sizes': [field_sizes, 30, 1],
        'layer_acts': ['none', 'none'],
        'layer_keeps': [1, 1],
        'opt_algo': 'adam',
        'learning_rate': 0.0001,
        'layer_l2': [0, 0],
        'kernel_l2': 0,
        'random_seed': 0,
        'has_field_bias': False
    })
d_name_model['fwfm_k_50_lr_0.0001'] = FwFM(**{
        'layer_sizes': [field_sizes, 50, 1],
        'layer_acts': ['none', 'none'],
        'layer_keeps': [1, 1],
        'opt_algo': 'adam',
        'learning_rate': 0.0001,
        'layer_l2': [0, 0],
        'kernel_l2': 0,
        'random_seed': 0,
        'has_field_bias': False
    })
d_name_model['fwfm_k_100_lr_0.0001'] = FwFM(**{
        'layer_sizes': [field_sizes, 100, 1],
        'layer_acts': ['none', 'none'],
        'layer_keeps': [1, 1],
        'opt_algo': 'adam',
        'learning_rate': 0.0001,
        'layer_l2': [0, 0],
        'kernel_l2': 0,
        'random_seed': 0,
        'has_field_bias': False
    })
d_name_model['fwfm_k_200_lr_0.0001'] = FwFM(**{
        'layer_sizes': [field_sizes, 200, 1],
        'layer_acts': ['none', 'none'],
        'layer_keeps': [1, 1],
        'opt_algo': 'adam',
        'learning_rate': 0.0001,
        'layer_l2': [0, 0],
        'kernel_l2': 0,
        'random_seed': 0,
        'has_field_bias': False
    })
'''
d_name_model['fwfm_le'] = FwFM_LE(**{
        'layer_sizes': [field_sizes, 10, 1],
        'layer_acts': ['tanh', 'none'],
        'layer_keeps': [1, 1],
        'opt_algo': 'adam',
        'learning_rate': 0.0005,
        'layer_l2': [0, 0],
        'kernel_l2': 0,
        'random_seed': 0,
        'has_field_bias': False
    })
'''
d_name_model['fwfm_with_field_bias'] = FwFM(**{
        'layer_sizes': [field_sizes, 10, 1],
        'layer_acts': ['tanh', 'none'],
        'layer_keeps': [1, 1],
        'opt_algo': 'adam',
        'learning_rate': 0.0005,
        'layer_l2': [0, 0],
        'kernel_l2': 0,
        'random_seed': 0,
        'has_field_bias': True
    })
d_name_model['fwfm_0.0001_without_field_bias'] = FwFM(**{
        'layer_sizes': [field_sizes, 10, 1],
        'layer_acts': ['tanh', 'none'],
        'layer_keeps': [1, 1],
        'opt_algo': 'adam',
        'learning_rate': 0.0001,
        'layer_l2': [0, 0],
        'kernel_l2': 0,
        'random_seed': 0,
        'has_field_bias': False
    })
d_name_model['fwfm_0.0005_without_field_bias_20'] = FwFM(**{
        'layer_sizes': [field_sizes, 20, 1],
        'layer_acts': ['tanh', 'none'],
        'layer_keeps': [1, 1],
        'opt_algo': 'adam',
        'learning_rate': 0.0005,
        'layer_l2': [0, 0],
        'kernel_l2': 0,
        'random_seed': 0,
        'has_field_bias': False
    })
d_name_model['fwfm_r_l2_0.1'] = FwFM(**{
    'layer_sizes': [field_sizes, 10, 1],
    'layer_acts': ['none', 'none'],
    'layer_keeps': [1, 1],
    'opt_algo': 'adam',
    'learning_rate': 0.0005,
    'layer_l2': [0, 0],
    'kernel_l2': 0,
    'random_seed': 0,
    'l2_dict': {
        'linear_w': 0.0,
        'v': 0.00,
        'r': 0.1
    }
})
d_name_model['fwfm_r_l2_0.01'] = FwFM(**{
    'layer_sizes': [field_sizes, 10, 1],
    'layer_acts': ['none', 'none'],
    'layer_keeps': [1, 1],
    'opt_algo': 'adam',
    'learning_rate': 0.0005,
    'layer_l2': [0, 0],
    'kernel_l2': 0,
    'random_seed': 0,
    'l2_dict': {
        'linear_w': 0.0,
        'v': 0.00,
        'r': 0.01
    }
})
d_name_model['fwfm_r_l2_0.001'] = FwFM(**{
    'layer_sizes': [field_sizes, 10, 1],
    'layer_acts': ['none', 'none'],
    'layer_keeps': [1, 1],
    'opt_algo': 'adam',
    'learning_rate': 0.0005,
    'layer_l2': [0, 0],
    'kernel_l2': 0,
    'random_seed': 0,
    'l2_dict': {
        'linear_w': 0.0,
        'v': 0.00,
        'r': 0.001
    }
})
d_name_model['fwfm_r_l2_0.0001'] = FwFM(**{
    'layer_sizes': [field_sizes, 10, 1],
    'layer_acts': ['none', 'none'],
    'layer_keeps': [1, 1],
    'opt_algo': 'adam',
    'learning_rate': 0.0005,
    'layer_l2': [0, 0],
    'kernel_l2': 0,
    'random_seed': 0,
    'l2_dict': {
        'linear_w': 0.0,
        'v': 0.00,
        'r': 0.0001
    }
})
d_name_model['fwfm_r_l2_0.00001'] = FwFM(**{
    'layer_sizes': [field_sizes, 10, 1],
    'layer_acts': ['none', 'none'],
    'layer_keeps': [1, 1],
    'opt_algo': 'adam',
    'learning_rate': 0.0005,
    'layer_l2': [0, 0],
    'kernel_l2': 0,
    'random_seed': 0,
    'l2_dict': {
        'linear_w': 0.0,
        'v': 0.00,
        'r': 0.00001
    }
})
d_name_model['fwfm_r_l2_0.000001'] = FwFM(**{
    'layer_sizes': [field_sizes, 10, 1],
    'layer_acts': ['none', 'none'],
    'layer_keeps': [1, 1],
    'opt_algo': 'adam',
    'learning_rate': 0.0005,
    'layer_l2': [0, 0],
    'kernel_l2': 0,
    'random_seed': 0,
    'l2_dict': {
        'linear_w': 0.0,
        'v': 0.00,
        'r': 0.000001
    }
})
d_name_model['fwfm_r_l2_0.0000001'] = FwFM(**{
    'layer_sizes': [field_sizes, 10, 1],
    'layer_acts': ['none', 'none'],
    'layer_keeps': [1, 1],
    'opt_algo': 'adam',
    'learning_rate': 0.0005,
    'layer_l2': [0, 0],
    'kernel_l2': 0,
    'random_seed': 0,
    'l2_dict': {
        'linear_w': 0.0,
        'v': 0.00,
        'r': 0.0000001
    }
})
d_name_model['fwfm_v_l2_0.1'] = FwFM(**{
    'layer_sizes': [field_sizes, 10, 1],
    'layer_acts': ['none', 'none'],
    'layer_keeps': [1, 1],
    'opt_algo': 'adam',
    'learning_rate': 0.0005,
    'layer_l2': [0, 0],
    'kernel_l2': 0,
    'random_seed': 0,
    'l2_dict': {
        'linear_w': 0.0,
        'v': 0.1,
        'r': 0.0
    }
})
d_name_model['fwfm_v_l2_0.01'] = FwFM(**{
    'layer_sizes': [field_sizes, 10, 1],
    'layer_acts': ['none', 'none'],
    'layer_keeps': [1, 1],
    'opt_algo': 'adam',
    'learning_rate': 0.0005,
    'layer_l2': [0, 0],
    'kernel_l2': 0,
    'random_seed': 0,
    'l2_dict': {
        'linear_w': 0.0,
        'v': 0.01,
        'r': 0.0
    }
})
d_name_model['fwfm_v_l2_0.001'] = FwFM(**{
    'layer_sizes': [field_sizes, 10, 1],
    'layer_acts': ['none', 'none'],
    'layer_keeps': [1, 1],
    'opt_algo': 'adam',
    'learning_rate': 0.0005,
    'layer_l2': [0, 0],
    'kernel_l2': 0,
    'random_seed': 0,
    'l2_dict': {
        'linear_w': 0.0,
        'v': 0.001,
        'r': 0.0
    }
})
d_name_model['fwfm_v_l2_0.0001'] = FwFM(**{
    'layer_sizes': [field_sizes, 10, 1],
    'layer_acts': ['none', 'none'],
    'layer_keeps': [1, 1],
    'opt_algo': 'adam',
    'learning_rate': 0.0005,
    'layer_l2': [0, 0],
    'kernel_l2': 0,
    'random_seed': 0,
    'l2_dict': {
        'linear_w': 0.0,
        'v': 0.0001,
        'r': 0.0
    }
})
d_name_model['fwfm_v_l2_0.00001'] = FwFM(**{
    'layer_sizes': [field_sizes, 10, 1],
    'layer_acts': ['none', 'none'],
    'layer_keeps': [1, 1],
    'opt_algo': 'adam',
    'learning_rate': 0.0005,
    'layer_l2': [0, 0],
    'kernel_l2': 0,
    'random_seed': 0,
    'l2_dict': {
        'linear_w': 0.0,
        'v': 0.00001,
        'r': 0.0
    }
})
d_name_model['fwfm_v_l2_0.000001'] = FwFM(**{
    'layer_sizes': [field_sizes, 10, 1],
    'layer_acts': ['none', 'none'],
    'layer_keeps': [1, 1],
    'opt_algo': 'adam',
    'learning_rate': 0.0005,
    'layer_l2': [0, 0],
    'kernel_l2': 0,
    'random_seed': 0,
    'l2_dict': {
        'linear_w': 0.0,
        'v': 0.000001,
        'r': 0.0
    }
})
d_name_model['fwfm_v_l2_0.0000001'] = FwFM(**{
    'layer_sizes': [field_sizes, 10, 1],
    'layer_acts': ['none', 'none'],
    'layer_keeps': [1, 1],
    'opt_algo': 'adam',
    'learning_rate': 0.0005,
    'layer_l2': [0, 0],
    'kernel_l2': 0,
    'random_seed': 0,
    'l2_dict': {
        'linear_w': 0.0,
        'v': 0.0000001,
        'r': 0.0
    }
})
d_name_model['fwfm_w_l2_0.1'] = FwFM(**{
    'layer_sizes': [field_sizes, 10, 1],
    'layer_acts': ['none', 'none'],
    'layer_keeps': [1, 1],
    'opt_algo': 'adam',
    'learning_rate': 0.0005,
    'layer_l2': [0, 0],
    'kernel_l2': 0,
    'random_seed': 0,
    'l2_dict': {
        'linear_w': 0.1,
        'v': 0.0,
        'r': 0.0
    }
})
d_name_model['fwfm_w_l2_0.01'] = FwFM(**{
    'layer_sizes': [field_sizes, 10, 1],
    'layer_acts': ['none', 'none'],
    'layer_keeps': [1, 1],
    'opt_algo': 'adam',
    'learning_rate': 0.0005,
    'layer_l2': [0, 0],
    'kernel_l2': 0,
    'random_seed': 0,
    'l2_dict': {
        'linear_w': 0.01,
        'v': 0.0,
        'r': 0.0
    }
})
d_name_model['fwfm_w_l2_0.001'] = FwFM(**{
    'layer_sizes': [field_sizes, 10, 1],
    'layer_acts': ['none', 'none'],
    'layer_keeps': [1, 1],
    'opt_algo': 'adam',
    'learning_rate': 0.0005,
    'layer_l2': [0, 0],
    'kernel_l2': 0,
    'random_seed': 0,
    'l2_dict': {
        'linear_w': 0.001,
        'v': 0.0,
        'r': 0.0
    }
})
d_name_model['fwfm_w_l2_0.0001'] = FwFM(**{
    'layer_sizes': [field_sizes, 10, 1],
    'layer_acts': ['none', 'none'],
    'layer_keeps': [1, 1],
    'opt_algo': 'adam',
    'learning_rate': 0.0005,
    'layer_l2': [0, 0],
    'kernel_l2': 0,
    'random_seed': 0,
    'l2_dict': {
        'linear_w': 0.0001,
        'v': 0.0,
        'r': 0.0
    }
})
d_name_model['fwfm_w_l2_0.00001'] = FwFM(**{
    'layer_sizes': [field_sizes, 10, 1],
    'layer_acts': ['none', 'none'],
    'layer_keeps': [1, 1],
    'opt_algo': 'adam',
    'learning_rate': 0.0005,
    'layer_l2': [0, 0],
    'kernel_l2': 0,
    'random_seed': 0,
    'l2_dict': {
        'linear_w': 0.00001,
        'v': 0.0,
        'r': 0.0
    }
})
d_name_model['fwfm_w_l2_0.000001'] = FwFM(**{
    'layer_sizes': [field_sizes, 10, 1],
    'layer_acts': ['none', 'none'],
    'layer_keeps': [1, 1],
    'opt_algo': 'adam',
    'learning_rate': 0.0005,
    'layer_l2': [0, 0],
    'kernel_l2': 0,
    'random_seed': 0,
    'l2_dict': {
        'linear_w': 0.000001,
        'v': 0.0,
        'r': 0.0
    }
})
d_name_model['fwfm_gd'] = FwFM(**{
        'layer_sizes': [field_sizes, 10, 1],
        'layer_acts': ['none', 'none'],
        'layer_keeps': [1, 1],
        'opt_algo': 'gd',
        'learning_rate': 0.0005,
        'layer_l2': [0, 0],
        'kernel_l2': 0,
        'random_seed': 0
    })
d_name_model['fwfm_momentum'] = FwFM(**{
        'layer_sizes': [field_sizes, 10, 1],
        'layer_acts': ['none', 'none'],
        'layer_keeps': [1, 1],
        'opt_algo': 'momentum',
        'learning_rate': 0.0005,
        'layer_l2': [0, 0],
        'kernel_l2': 0,
        'random_seed': 0
    })
d_name_model['fwfm_nesterov'] = FwFM(**{
        'layer_sizes': [field_sizes, 10, 1],
        'layer_acts': ['none', 'none'],
        'layer_keeps': [1, 1],
        'opt_algo': 'nesterov',
        'learning_rate': 0.0005,
        'layer_l2': [0, 0],
        'kernel_l2': 0,
        'random_seed': 0
    })
d_name_model['fwfm_adagrad'] = FwFM(**{
        'layer_sizes': [field_sizes, 10, 1],
        'layer_acts': ['none', 'none'],
        'layer_keeps': [1, 1],
        'opt_algo': 'adagrad',
        'learning_rate': 0.0005,
        'layer_l2': [0, 0],
        'kernel_l2': 0,
        'random_seed': 0
    })
d_name_model['fwfm_adadelta'] = FwFM(**{
        'layer_sizes': [field_sizes, 10, 1],
        'layer_acts': ['none', 'none'],
        'layer_keeps': [1, 1],
        'opt_algo': 'adadelta',
        'learning_rate': 0.0005,
        'layer_l2': [0, 0],
        'kernel_l2': 0,
        'random_seed': 0
    })
d_name_model['fwfm_adam'] = FwFM(**{
        'layer_sizes': [field_sizes, 10, 1],
        'layer_acts': ['none', 'none'],
        'layer_keeps': [1, 1],
        'opt_algo': 'adam',
        'learning_rate': 0.0005,
        'layer_l2': [0, 0],
        'kernel_l2': 0,
        'random_seed': 0
    })
#for name in d_name_model.keys():
#for name in ['fast_ctr_concat', 'fnn']:
#for name in ['pnn1_fixed_0.001', 'pnn1_fixed_0.001_5', 'pnn1_fixed_0.001_20', 'pnn1_fixed_0.001_50', 'pnn1_fixed_0.001_gd']:
#for name in ['pnn1_fixed_0.001', 'pnn1_fixed_0.001_5', 'pnn1_fixed_0.001_20', 'pnn1_fixed_0.001_50', 'pnn1_fixed_0.001_gd', 'pnn1_fixed_0.001_dropout-0.5', 'pnn1_fixed_0.001_l2-1-0.5']:
#for name in ['pnn1_fixed_0.001_20', 'pnn1_fixed_0.001_50', 'pnn1_fixed_0.001_gd', 'pnn1_fixed_0.001_dropout-0.5', 'pnn1_fixed_0.001_l2-1-0.5']:
#for name in ['pnn1', 'pnn2', 'pnn1_fixed', 'pnn1_fixed_0.001']:
#for name in ['pnn1_fixed_0.00001']:
#for name in ['fm']:
#for name in ['pnn1_0.0005', 'pnn2_0.0005', 'pnn1_fixed_0.0005']:
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
for name in ['fwfm']:
    print 'name with none activation', name
    sys.stdout.flush()
    model = d_name_model[name]
    train(model, 'yahoo_dataset2.2_' + name)
