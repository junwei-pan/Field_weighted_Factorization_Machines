import utils
field_sizes = utils.FIELD_SIZES
d_name_conf = utils.d_name_conf

d_name_conf['fwfm'] = {
        'layer_sizes': [field_sizes, 10, 1],
        'layer_acts': ['none', 'none'],
        'layer_keeps': [1, 1],
        'opt_algo': 'adam',
        'learning_rate': 5e-4,
        'layer_l2': [0, 0],
        'kernel_l2': 0,
        'random_seed': 0,
        'l2_dict': {
            'linear_w': 0.0,
            'v': 0.00,
            'r': 0.1
        }
    }
d_name_conf['fwfm_l2_v_1e-1'] = {
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
        'v': 0.1,
        'r': 0.0
    }
}
d_name_conf['fwfm_l2_v_1e-2'] = {
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
        'v': 1e-2,
        'r': 0.0
    }
}
d_name_conf['fwfm_l2_v_1e-3'] = {
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
        'v': 0.0,
        'r': 1e-3
    }
}
d_name_conf['fwfm_l2_v_1e-4'] = {
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
        'v': 1e-4,
        'r': 0.0
    }
}
d_name_conf['fwfm_l2_v_1e-5'] = {
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
    }
}
d_name_conf['fwfm_l2_v_1e-6'] = {
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
        'v': 1e-6,
        'r': 0.0
    }
}
d_name_conf['fwfm_l2_v_1e-7'] = {
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
        'v': 1e-7,
        'r': 0.0
    }
}
d_name_conf['fwfm_l2_v_1e-8'] = {
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
        'v': 1e-8,
        'r': 0.0
    }
}
d_name_conf['fwfm_l2_v_1e-5_lr_1e-2'] = {
    'layer_sizes': [field_sizes, 10, 1],
    'layer_acts': ['none', 'none'],
    'layer_keeps': [1, 1],
    'opt_algo': 'adam',
    'learning_rate': 1e-2,
    'layer_l2': [0, 0],
    'kernel_l2': 0,
    'random_seed': 0,
    'l2_dict': {
        'linear_w': 0.0,
        'v': 1e-5,
        'r': 0.0
    }
}
d_name_conf['fwfm_l2_v_1e-5_lr_1e-3'] = {
    'layer_sizes': [field_sizes, 10, 1],
    'layer_acts': ['none', 'none'],
    'layer_keeps': [1, 1],
    'opt_algo': 'adam',
    'learning_rate': 1e-3,
    'layer_l2': [0, 0],
    'kernel_l2': 0,
    'random_seed': 0,
    'l2_dict': {
        'linear_w': 0.0,
        'v': 1e-5,
        'r': 0.0
    }
}
d_name_conf['fwfm_l2_v_1e-5_lr_5e-4'] = {
    'layer_sizes': [field_sizes, 10, 1],
    'layer_acts': ['none', 'none'],
    'layer_keeps': [1, 1],
    'opt_algo': 'adam',
    'learning_rate': 5e-4,
    'layer_l2': [0, 0],
    'kernel_l2': 0,
    'random_seed': 0,
    'l2_dict': {
        'linear_w': 0.0,
        'v': 1e-5,
        'r': 0.0
    }
}
d_name_conf['fwfm_l2_v_1e-5_lr_5e-5'] = {
    'layer_sizes': [field_sizes, 10, 1],
    'layer_acts': ['none', 'none'],
    'layer_keeps': [1, 1],
    'opt_algo': 'adam',
    'learning_rate': 5e-5,
    'layer_l2': [0, 0],
    'kernel_l2': 0,
    'random_seed': 0,
    'l2_dict': {
        'linear_w': 0.0,
        'v': 1e-5,
        'r': 0.0
    }
}
d_name_conf['fwfm_l2_v_1e-5_lr_1e-5'] = {
    'layer_sizes': [field_sizes, 10, 1],
    'layer_acts': ['none', 'none'],
    'layer_keeps': [1, 1],
    'opt_algo': 'adam',
    'learning_rate': 1e-5,
    'layer_l2': [0, 0],
    'kernel_l2': 0,
    'random_seed': 0,
    'l2_dict': {
        'linear_w': 0.0,
        'v': 1e-5,
        'r': 0.0
    }
}
d_name_conf['fwfm_l2_v_1e-5_lr_1e-6'] = {
    'layer_sizes': [field_sizes, 10, 1],
    'layer_acts': ['none', 'none'],
    'layer_keeps': [1, 1],
    'opt_algo': 'adam',
    'learning_rate': 1e-6,
    'layer_l2': [0, 0],
    'kernel_l2': 0,
    'random_seed': 0,
    'l2_dict': {
        'linear_w': 0.0,
        'v': 1e-5,
        'r': 0.0
    }
}
d_name_conf['fwfm_l2_v_1e-5_lr_1e-7'] = {
    'layer_sizes': [field_sizes, 10, 1],
    'layer_acts': ['none', 'none'],
    'layer_keeps': [1, 1],
    'opt_algo': 'adam',
    'learning_rate': 1e-7,
    'layer_l2': [0, 0],
    'kernel_l2': 0,
    'random_seed': 0,
    'l2_dict': {
        'linear_w': 0.0,
        'v': 1e-5,
        'r': 0.0
    }
}

