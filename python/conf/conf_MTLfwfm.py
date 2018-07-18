import utils
field_sizes = utils.FIELD_SIZES
d_name_conf = utils.d_name_conf

d_name_conf['MTLfwfm_l2_v_1e-5'] = {
    'layer_sizes': [field_sizes, 10, 1],
    'layer_acts': ['none', 'none'],
    'layer_keeps': [1, 1],
    'opt_algo': 'adam',
    'learning_rate': 0.0001,
    'layer_l2': [0, 0],
    'kernel_l2': 0,
    'random_seed': 0,
    'l2_dict': {
        'linear_w': 0.0,
        'v': 0.00001,
        'r': 0.0
    },
    'index_lines': -1,
    'num_lines': -1
}