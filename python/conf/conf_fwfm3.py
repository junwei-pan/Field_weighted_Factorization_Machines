import utils
field_sizes = utils.FIELD_SIZES
d_name_conf = utils.d_name_conf

d_name_conf['fwfm3'] = {
        'layer_sizes': [field_sizes, 10, 1],
        'layer_acts': ['none', 'none'],
        'layer_keeps': [1, 1],
        'opt_algo': 'adam',
        'learning_rate': 5e-4,
        'layer_l2': [0, 0],
        'kernel_l2': 0,
        'random_seed': 0,
        'l2_dict': {
            'v': 0.1
        }
    }

d_name_conf['fwfm3_15'] = {
    'layer_sizes': [field_sizes, 10, 1],
    'layer_acts': ['none', 'none'],
    'layer_keeps': [1, 1],
    'opt_algo': 'adam',
    'learning_rate': 1e-4,
    'layer_l2': [0, 0],
    'kernel_l2': 0,
    'random_seed': 0,
    'fullLayer3' : False, 
    'survivors' : 15,
    'l2_dict': {
        'v': 0.1
    }
}