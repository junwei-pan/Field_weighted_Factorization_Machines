import utils
field_sizes = utils.FIELD_SIZES
d_name_conf = utils.d_name_conf

# Tune the L2 for embedding vectors.
d_name_conf['ffm_l2_v_1e-1'] = {
        'layer_sizes': [field_sizes, 10, 1],
        'layer_acts': ['none', 'none'],
        'layer_keeps': [1, 1],
        'opt_algo': 'adam',
        'learning_rate': 0.0001,
        'layer_l2': [0, 0],
        'kernel_l2': 0,
        'random_seed': 0,
        'l2_dict': {
            'v': 0.1,
        }
    }

d_name_conf['ffm_l2_v_1e-2'] = {
        'layer_sizes': [field_sizes, 10, 1],
        'layer_acts': ['none', 'none'],
        'layer_keeps': [1, 1],
        'opt_algo': 'adam',
        'learning_rate': 0.0001,
        'layer_l2': [0, 0],
        'kernel_l2': 0,
        'random_seed': 0,
        'l2_dict': {
            'v': 0.01,
        }
    }

d_name_conf['ffm_l2_v_1e-3'] = {
        'layer_sizes': [field_sizes, 10, 1],
        'layer_acts': ['none', 'none'],
        'layer_keeps': [1, 1],
        'opt_algo': 'adam',
        'learning_rate': 0.0001,
        'layer_l2': [0, 0],
        'kernel_l2': 0,
        'random_seed': 0,
        'l2_dict': {
            'v': 0.001,
        }
    }
d_name_conf['ffm_l2_v_1e-4'] = {
        'layer_sizes': [field_sizes, 10, 1],
        'layer_acts': ['none', 'none'],
        'layer_keeps': [1, 1],
        'opt_algo': 'adam',
        'learning_rate': 0.0001,
        'layer_l2': [0, 0],
        'kernel_l2': 0,
        'random_seed': 0,
        'l2_dict': {
            'v': 0.0001,
        }
    }
d_name_conf['ffm_l2_v_1e-5'] = {
        'layer_sizes': [field_sizes, 10, 1],
        'layer_acts': ['none', 'none'],
        'layer_keeps': [1, 1],
        'opt_algo': 'adam',
        'learning_rate': 0.0001,
        'layer_l2': [0, 0],
        'kernel_l2': 0,
        'random_seed': 0,
        'l2_dict': {
            'v': 0.00001,
        }
    }
d_name_conf['ffm_l2_v_1e-6'] = {
        'layer_sizes': [field_sizes, 10, 1],
        'layer_acts': ['none', 'none'],
        'layer_keeps': [1, 1],
        'opt_algo': 'adam',
        'learning_rate': 0.0001,
        'layer_l2': [0, 0],
        'kernel_l2': 0,
        'random_seed': 0,
        'l2_dict': {
            'v': 0.000001,
        }
    }
d_name_conf['ffm_l2_v_1e-7'] = {
        'layer_sizes': [field_sizes, 10, 1],
        'layer_acts': ['none', 'none'],
        'layer_keeps': [1, 1],
        'opt_algo': 'adam',
        'learning_rate': 0.0001,
        'layer_l2': [0, 0],
        'kernel_l2': 0,
        'random_seed': 0,
        'l2_dict': {
            'v': 0.0000001,
        }
    }
d_name_conf['ffm_l2_v_1e-8'] = {
        'layer_sizes': [field_sizes, 10, 1],
        'layer_acts': ['none', 'none'],
        'layer_keeps': [1, 1],
        'opt_algo': 'adam',
        'learning_rate': 0.0001,
        'layer_l2': [0, 0],
        'kernel_l2': 0,
        'random_seed': 0,
        'l2_dict': {
            'v': 0.00000001,
        }
    }
d_name_conf['ffm_l2_v_0'] = {
        'layer_sizes': [field_sizes, 10, 1],
        'layer_acts': ['none', 'none'],
        'layer_keeps': [1, 1],
        'opt_algo': 'adam',
        'learning_rate': 0.0001,
        'layer_l2': [0, 0],
        'kernel_l2': 0,
        'random_seed': 0,
        'l2_dict': {
            'v': 0.0,
        }
    }
d_name_conf['ffm_l2_v_1e-7_lr_1e-1'] = {
        'layer_sizes': [field_sizes, 10, 1],
        'layer_acts': ['none', 'none'],
        'layer_keeps': [1, 1],
        'opt_algo': 'adam',
        'learning_rate': 0.1,
        'layer_l2': [0, 0],
        'kernel_l2': 0,
        'random_seed': 0,
        'l2_dict': {
            'v': 0.0000001,
        }
    }
d_name_conf['ffm_l2_v_1e-7_lr_1e-2'] = {
        'layer_sizes': [field_sizes, 10, 1],
        'layer_acts': ['none', 'none'],
        'layer_keeps': [1, 1],
        'opt_algo': 'adam',
        'learning_rate': 0.01,
        'layer_l2': [0, 0],
        'kernel_l2': 0,
        'random_seed': 0,
        'l2_dict': {
            'v': 0.0000001,
        }
    }
d_name_conf['ffm_l2_v_1e-7_lr_1e-3'] = {
        'layer_sizes': [field_sizes, 10, 1],
        'layer_acts': ['none', 'none'],
        'layer_keeps': [1, 1],
        'opt_algo': 'adam',
        'learning_rate': 0.001,
        'layer_l2': [0, 0],
        'kernel_l2': 0,
        'random_seed': 0,
        'l2_dict': {
            'v': 0.0000001,
        }
    }
d_name_conf['ffm_l2_v_1e-7_lr_1e-4'] = {
        'layer_sizes': [field_sizes, 10, 1],
        'layer_acts': ['none', 'none'],
        'layer_keeps': [1, 1],
        'opt_algo': 'adam',
        'learning_rate': 0.0001,
        'layer_l2': [0, 0],
        'kernel_l2': 0,
        'random_seed': 0,
        'l2_dict': {
            'v': 0.0000001,
        }
    }
d_name_conf['ffm_l2_v_1e-7_lr_1e-5'] = {
        'layer_sizes': [field_sizes, 10, 1],
        'layer_acts': ['none', 'none'],
        'layer_keeps': [1, 1],
        'opt_algo': 'adam',
        'learning_rate': 0.00001,
        'layer_l2': [0, 0],
        'kernel_l2': 0,
        'random_seed': 0,
        'l2_dict': {
            'v': 0.0000001,
        }
    }
d_name_conf['ffm_l2_v_1e-7_lr_1e-6'] = {
        'layer_sizes': [field_sizes, 10, 1],
        'layer_acts': ['none', 'none'],
        'layer_keeps': [1, 1],
        'opt_algo': 'adam',
        'learning_rate': 0.000001,
        'layer_l2': [0, 0],
        'kernel_l2': 0,
        'random_seed': 0,
        'l2_dict': {
            0.0000001,
        }
    }
