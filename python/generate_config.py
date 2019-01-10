import configparser

# CVR, Yahoo
config = configparser.ConfigParser()
config['setup'] = {'path_train': '../data/data_cvr/cvr_imp_20180704_0710_conv_20180704_0716.csv.add_conv_type.thres5.yx',
                   'path_validation': '../data/data_cvr/cvr_imp_20180711_conv_20180711_0717.csv.add_conv_type.thres5.yx',
                   'path_test': '../data/data_cvr/cvr_imp_20180712_conv_20180712_0718.csv.add_conv_type.thres5.yx',
                   'path_feature_index': '../data/data_cvr/featureindex_thres5.txt',
                   'num_field': 18
                   }

with open('conf/project/cvr.ini', 'w') as configfile:
    config.write(configfile)

# CVR, Yahoo
config = configparser.ConfigParser()
config['setup'] = {'path_train': '../data/data_cvr_without_conv/cvr_imp_20180704_0710_conv_20180704_0716.csv.add_conv_type.remove_conv_type.thres5.yx',
                   'path_validation': '../data/data_cvr_without_conv/cvr_imp_20180711_conv_20180711_0717.csv.add_conv_type.remove_conv_type.thres5.yx',
                   'path_test': '../data/data_cvr_without_conv/cvr_imp_20180712_conv_20180712_0718.csv.add_conv_type.remove_conv_type.thres5.yx',
                   'path_feature_index': '../data/data_cvr_without_conv/featureindex_thres5.txt',
                   'num_field': 17
                   }

with open('conf/project/cvr_without_conv_type.ini', 'w') as configfile:
    config.write(configfile)

# CVR2, Yahoo
config = configparser.ConfigParser()
config['setup'] = {'path_train': '../data/data_cvr2/cvr_imp_20180708_0714_conv_20180708_0720.csv.add_conv_type.thres5.yx',
                   'path_validation': '../data/data_cvr2/cvr_imp_20180715_0721_conv_20180715_0727.csv.add_conv_type.thres5.yx',
                   'path_test': '../data/data_cvr2/cvr_imp_20180722_0728_conv_20180722_0803.csv.add_conv_type.thres5.yx',
                   'path_feature_index': '../data/data_cvr2/featureindex_thres5.txt',
                   'num_field': 17
                   }

with open('conf/project/cvr2.ini', 'w') as configfile:
    config.write(configfile)

# CVR2, only view_content in train
config = configparser.ConfigParser()
config['setup'] = {'path_train': '../data/data_cvr2/cvr_imp_20180708_0714_conv_20180708_0720.csv.add_conv_type.thres5.yx.View_Content',
                   'path_validation': '../data/data_cvr2/cvr_imp_20180715_0721_conv_20180715_0727.csv.add_conv_type.thres5.yx',
                   'path_test': '../data/data_cvr2/cvr_imp_20180722_0728_conv_20180722_0803.csv.add_conv_type.thres5.yx',
                   'path_feature_index': '../data/data_cvr2/featureindex_thres5.txt',
                   'num_field': 17
                   }

with open('conf/project/cvr2_view_content.ini', 'w') as configfile:
    config.write(configfile)

# CVR3, Yahoo, new hashing range
config = configparser.ConfigParser()
config['setup'] = {'path_train': '../data/data_cvr3/cvr_imp_20180708_0714_conv_20180708_0720.csv.add_conv_type.thres5.yx',
                   'path_validation': '../data/data_cvr3/cvr_imp_20180715_0721_conv_20180715_0727.csv.add_conv_type.thres5.yx',
                   'path_test': '../data/data_cvr3/cvr_imp_20180722_0728_conv_20180722_0803.csv.add_conv_type.thres5.yx',
                   'path_feature_index': '../data/data_cvr3/featureindex_thres5.txt',
                   'num_field': 17
                   }

with open('conf/project/cvr3.ini', 'w') as configfile:
    config.write(configfile)

# CVR3, Yahoo, new hashing range, rebalanced
config = configparser.ConfigParser()
config['setup'] = {'path_train': '../data/data_cvr3/cvr_imp_20180708_0714_conv_20180708_0720.csv.add_conv_type.thres5.yx.rebalanced',
                   'path_validation': '../data/data_cvr4/cvr_imp_20180715_0721_conv_20180715_0727.csv.add_conv_type.thres5.neg_sampling_0.25.yx',
                   'path_test': '../data/data_cvr4/cvr_imp_20180722_0728_conv_20180722_0803.csv.add_conv_type.thres5.neg_sampling_0.25.yx',
                   'path_feature_index': '../data/data_cvr3/featureindex_thres5.txt',
                   'num_field': 17
                   }

with open('conf/project/cvr3_rebalanced.ini', 'w') as configfile:
    config.write(configfile)

# CVR5, Yahoo, for validation and test, only downsample negative samples.
config = configparser.ConfigParser()
config['setup'] = {'path_train': '../data/data_cvr5/cvr_imp_20180808_0814_conv_20180808_0820.csv.add_conv_type.thres5.yx',
                   'path_validation': '../data/data_cvr5/cvr_imp_20180815_conv_20180815_0821.csv.add_conv_type.thres5.yx',
                   'path_test': '../data/data_cvr5/cvr_imp_20180816_conv_20180816_0822.csv.add_conv_type.thres5.yx',
                   'path_feature_index': '../data/data_cvr5/featureindex_thres5.txt',
                   'num_field': 17
                   }

with open('conf/project/cvr5.ini', 'w') as configfile:
    config.write(configfile)

# CVR5, Yahoo, for validation and test, only downsample negative samples.
config = configparser.ConfigParser()
config['setup'] = {'path_train': '../data/data_cvr5/cvr_imp_20180808_0814_conv_20180808_0820.csv.add_conv_type.thres1.yx',
                   'path_validation': '../data/data_cvr5/cvr_imp_20180815_conv_20180815_0821.csv.add_conv_type.thres1.yx',
                   'path_test': '../data/data_cvr5/cvr_imp_20180816_conv_20180816_0822.csv.add_conv_type.thres1.yx',
                   'path_feature_index': '../data/data_cvr5/featureindex_thres1.txt',
                   'num_field': 17
                   }

with open('conf/project/cvr5_thres1.ini', 'w') as configfile:
    config.write(configfile)

# CVR, Yahoo
config = configparser.ConfigParser()
config['setup'] = {'path_train': '../data_cvr/cvr_imp_20180704_0710_conv_20180704_0716.csv.add_conv_type.thres5.yx.10k',
                       'path_validation': '../data_cvr/cvr_imp_20180711_conv_20180711_0717.csv.add_conv_type.thres5.yx.10k',
                       'path_test': '../data_cvr/cvr_imp_20180712_conv_20180712_0718.csv.add_conv_type.thres5.yx.10k',
                       'path_feature_index': '../data_cvr/featureindex_thres5.txt',
                       'num_field': 18
                    }

with open('conf/project/cvr_10k.ini', 'w') as configfile:
    config.write(configfile)

# CVR6
config = configparser.ConfigParser()
path_dir = '../data/data_cvr6/'
config['setup'] = {'path_train': path_dir + 'cvr_imp_20181010_1016_conv_20181010_1022.csv.add_conv_type.thres5.yx',
                   'path_validation': path_dir + 'cvr_imp_20181017_conv_20181017_1023.csv.add_conv_type.thres5.yx',
                   'path_test': path_dir + 'cvr_imp_20181018_conv_20181018_1024.csv.add_conv_type.thres5.yx',
                   'path_feature_index': path_dir + 'featureindex_thres5.txt',
                   'num_field': 17
                   }

with open('conf/project/cvr6_thres5.ini', 'w') as configfile:
    config.write(configfile)

# Click, Yahoo
config = configparser.ConfigParser()
config['setup'] = {'path_train': '../data_yahoo/dataset2/ctr_20170517_0530_0.015.txt.thres10.yx',
                       'path_validation': '../data_yahoo/dataset2/ctr_20170531.txt.downsample_all.0.1.thres10.yx',
                       'path_test': '../data_yahoo/dataset2/ctr_20170601.txt.downsample_all.0.1.thres10.yx',
                       'path_feature_index': '../data_yahoo/dataset2/featindex_25m_thres10.txt',
                       'num_field': 15
                       }

with open('conf/project/click_yahoo_dataset2.ini', 'w') as configfile:
    config.write(configfile)
