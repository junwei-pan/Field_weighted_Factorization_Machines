import utils
field_sizes = utils.FIELD_SIZES
d_name_conf = utils.d_name_conf

conf_default = {
    'layer_sizes': [field_sizes, 10, 1],
    'layer_acts': ['none', 'none'],
    'layer_keeps': [1, 1],
    'opt_algo': 'adam',
    'learning_rate': 1e-4,
    'layer_l2': [0, 0],
    'kernel_l2': 0,
    'random_seed': 0,
    'l2_dict': {
        'linear_w': 0.0,
        'v': 1e-5,
        'r': 0.0
    },
    'index_lines': -1,
    'num_lines': -1
}

conf = conf_default.copy()
d_name_conf['MTLfwfm_l2_v_1e-5'] =  conf

conf = conf_default.copy()
conf['flag_r_factorized'] = True
conf['learning_rate'] = 5e-5
d_name_conf['MTLfwfm_r_factorized_lr_5e-5'] =  conf

conf = conf_default.copy()
conf['flag_r_factorized'] = True
conf['learning_rate'] = 1e-5
d_name_conf['MTLfwfm_r_factorized_lr_1e-5'] =  conf

conf = conf_default.copy()
conf['flag_r_factorized'] = True
conf['learning_rate'] = 5e-5
conf['l2_dict']['r'] = 1e-5
d_name_conf['MTLfwfm_r_factorized_lr_5e-5_l2_r_1e-5'] =  conf

conf = conf_default.copy()
conf['learning_rate'] = 5e-5
conf['l2_dict']['v'] = 5e-5
d_name_conf['MTLfwfm_lr_5e-5_l2_v_5e-5'] = conf

conf = conf_default.copy()
conf['learning_rate'] = 5e-5
conf['l2_dict']['v'] = 1e-6
d_name_conf['MTLfwfm_lr_5e-5_l2_v_1e-6'] = conf

conf = conf_default.copy()
conf['learning_rate'] = 5e-5
conf['l2_dict']['v'] = 1e-5
conf['l2_dict']['r'] = 1e-5
d_name_conf['MTLfwfm_lr_5e-5_l2_v_1e-5_l2_r_1e-5'] = conf

conf = conf_default.copy()
conf['learning_rate'] = 5e-5
d_name_conf['MTLfwfm_lr_5e-5_l2_v_1e-5'] = conf

