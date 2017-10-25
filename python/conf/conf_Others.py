
d_name_conf['lr'] = LR(**{
        'input_dim': input_dim,
        'opt_algo': 'adam',
        'learning_rate': 0.0001,
        'l2_weight': 0,
        'random_seed': 0
    })
d_name_conf['fm'] = FM(**{
        'input_dim': input_dim,
        'factor_order': 10,
        'opt_algo': 'adam',
        'learning_rate': 0.0001,
        'l2_w': 0,
        'l2_v': 0,
    })
d_name_conf['fnn'] = FNN(**{
        'layer_sizes': [field_sizes, 10, 1],
        'layer_acts': ['tanh', 'none'],
        'layer_keeps': [1, 1],
        'opt_algo': 'adam',
        'learning_rate': 0.0001,
        'layer_l2': [0, 0],
        'random_seed': 0
    })
d_name_conf['pnn1'] = PNN1(**{
        'layer_sizes': [field_sizes, 10, 1],
        'layer_acts': ['tanh', 'none'],
        'layer_keeps': [1, 1],
        'opt_algo': 'adam',
        'learning_rate': 0.0005,
        'layer_l2': [0, 0],
        'kernel_l2': 0,
        'random_seed': 0
    })
d_name_conf['pnn2'] = {
        'layer_sizes': [field_sizes, 10, 1],
        'layer_acts': ['tanh', 'none'],
        'layer_keeps': [1, 1],
        'opt_algo': 'adam',
        'learning_rate': 0.0005,
        'layer_l2': [0, 0],
        'kernel_l2': 0,
        'random_seed': 0
    }
d_name_conf['fast_ctr'] = {
        'layer_sizes': [field_sizes, 10, 1],
        'layer_acts': ['tanh', 'none'],
        'layer_keeps': [1, 1],
        'opt_algo': 'adam',
        'learning_rate': 0.0001,
        'layer_l2': [0, 0],
        'kernel_l2': 0,
        'random_seed': 0
}
d_name_conf['ffm'] = {
        'layer_sizes': [field_sizes, 10, 1],
        'layer_acts': ['none', 'none'],
        'layer_keeps': [1, 1],
        'opt_algo': 'adam',
        'learning_rate': 0.0005,
        'layer_l2': [0, 0],
        'kernel_l2': 0,
        'random_seed': 0,
    }
