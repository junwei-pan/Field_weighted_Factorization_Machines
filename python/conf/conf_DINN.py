import utils
field_sizes = utils.FIELD_SIZES
d_name_conf = utils.d_name_conf
num_layers = 5
arrayNumDI = [0, 0, 105, 40, 20]
arrayNumSurv = [0, 0, 20, 10]
conf_default = {
    'layer_sizes': [field_sizes, 10, 1, num_layers, arrayNumDI, arrayNumSurv],
    'allLayer2': True,
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
}

d_name_conf['DINN_lr_1e-4_l2_v_1e-5'] =  conf_default.copy()

