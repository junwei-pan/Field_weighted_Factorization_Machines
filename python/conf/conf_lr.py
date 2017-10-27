import utils
field_sizes = utils.FIELD_SIZES
d_name_conf = utils.d_name_conf
input_dim = utils.INPUT_DIM

# Tune the learning rate
d_name_conf['lr_l2_1e-7_lr_1e-1'] = {
        'input_dim': input_dim,
        'opt_algo': 'adam',
        'learning_rate': 0.1,
        'l2_weight': 0.0000001,
        'random_seed': 0
    }
d_name_conf['lr_l2_1e-7_lr_1e-2'] = {
        'input_dim': input_dim,
        'opt_algo': 'adam',
        'learning_rate': 0.01,
        'l2_weight': 0.0000001,
        'random_seed': 0
    }
d_name_conf['lr_l2_1e-7_lr_1e-3'] = {
        'input_dim': input_dim,
        'opt_algo': 'adam',
        'learning_rate': 0.001,
        'l2_weight': 0.0000001,
        'random_seed': 0
    }
d_name_conf['lr_l2_1e-7_lr_1e-5'] = {
        'input_dim': input_dim,
        'opt_algo': 'adam',
        'learning_rate': 0.00001,
        'l2_weight': 0.0000001,
        'random_seed': 0
    }
d_name_conf['lr_l2_1e-7_lr_1e-6'] = {
        'input_dim': input_dim,
        'opt_algo': 'adam',
        'learning_rate': 0.000001,
        'l2_weight': 0.0000001,
        'random_seed': 0
    }
d_name_conf['lr_l2_1e-7_lr_1e-7'] = {
        'input_dim': input_dim,
        'opt_algo': 'adam',
        'learning_rate': 0.0000001,
        'l2_weight': 0.0000001,
        'random_seed': 0
    }
d_name_conf['lr_l2_1e-7_lr_1e-8'] = {
        'input_dim': input_dim,
        'opt_algo': 'adam',
        'learning_rate': 0.00000001,
        'l2_weight': 0.0000001,
        'random_seed': 0
    }

# Tune the L2 for embedding vectors.
d_name_conf['lr_l2_1e-1'] = {
        'input_dim': input_dim,
        'opt_algo': 'adam',
        'learning_rate': 0.0001,
        'l2_weight': 0.1,
        'random_seed': 0
    }
d_name_conf['lr_l2_1e-2'] = {
        'input_dim': input_dim,
        'opt_algo': 'adam',
        'learning_rate': 0.0001,
        'l2_weight': 0.01,
        'random_seed': 0
    }
d_name_conf['lr_l2_1e-3'] = {
        'input_dim': input_dim,
        'opt_algo': 'adam',
        'learning_rate': 0.0001,
        'l2_weight': 0.001,
        'random_seed': 0
    }
d_name_conf['lr_l2_1e-4'] = {
        'input_dim': input_dim,
        'opt_algo': 'adam',
        'learning_rate': 0.0001,
        'l2_weight': 0.0001,
        'random_seed': 0
    }
d_name_conf['lr_l2_1e-5'] = {
        'input_dim': input_dim,
        'opt_algo': 'adam',
        'learning_rate': 0.0001,
        'l2_weight': 0.00001,
        'random_seed': 0
    }
d_name_conf['lr_l2_1e-6'] = {
        'input_dim': input_dim,
        'opt_algo': 'adam',
        'learning_rate': 0.0001,
        'l2_weight': 0.000001,
        'random_seed': 0
    }
d_name_conf['lr_l2_1e-7'] = {
        'input_dim': input_dim,
        'opt_algo': 'adam',
        'learning_rate': 0.0001,
        'l2_weight': 0.0000001,
        'random_seed': 0
    }
d_name_conf['lr_l2_1e-8'] = {
        'input_dim': input_dim,
        'opt_algo': 'adam',
        'learning_rate': 0.0001,
        'l2_weight': 0.00000001,
        'random_seed': 0
    }
d_name_conf['lr_l2_1e-9'] = {
        'input_dim': input_dim,
        'opt_algo': 'adam',
        'learning_rate': 0.0001,
        'l2_weight': 0.000000001,
        'random_seed': 0
    }


