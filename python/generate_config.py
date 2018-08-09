import configparser

# CVR, Yahoo
config = configparser.ConfigParser()
config['setup'] = {'path_train': '../data_cvr/cvr_imp_20180704_0710_conv_20180704_0716.csv.add_conv_type.thres5.yx',
                   'path_validation': '../data_cvr/cvr_imp_20180711_conv_20180711_0717.csv.add_conv_type.thres5.yx',
                   'path_test': '../data_cvr/cvr_imp_20180712_conv_20180712_0718.csv.add_conv_type.thres5.yx',
                   'path_feature_index': '../data_cvr/featureindex_thres5.txt',
                   'num_field': 18
                   }

with open('conf/project/cvr.ini', 'w') as configfile:
    config.write(configfile)

# CVR2, Yahoo
config = configparser.ConfigParser()
config['setup'] = {'path_train': '../data_cvr2/cvr_imp_20180708_0714_conv_20180708_0720.csv.add_conv_type.thres5.yx',
                   'path_validation': '../data_cvr2/cvr_imp_20180715_0721_conv_20180715_0727.csv.add_conv_type.thres5.yx',
                   'path_test': '../data_cvr2/cvr_imp_20180722_0728_conv_20180722_0803.csv.add_conv_type.thres5.yx',
                   'path_feature_index': '../data_cvr2/featureindex_thres5.txt',
                   'num_field': 17
                   }

with open('conf/project/cvr2.ini', 'w') as configfile:
    config.write(configfile)

# CVR2, only view_content in train
config = configparser.ConfigParser()
config['setup'] = {'path_train': '../data_cvr2/cvr_imp_20180708_0714_conv_20180708_0720.csv.add_conv_type.thres5.yx.View_Content',
                   'path_validation': '../data_cvr2/cvr_imp_20180715_0721_conv_20180715_0727.csv.add_conv_type.thres5.yx',
                   'path_test': '../data_cvr2/cvr_imp_20180722_0728_conv_20180722_0803.csv.add_conv_type.thres5.yx',
                   'path_feature_index': '../data_cvr2/featureindex_thres5.txt',
                   'num_field': 17
                   }

with open('conf/project/cvr2_view_content.ini', 'w') as configfile:
    config.write(configfile)

# CVR3, Yahoo, new hashing range
config = configparser.ConfigParser()
config['setup'] = {'path_train': '../data_cvr3/cvr_imp_20180708_0714_conv_20180708_0720.csv.add_conv_type.thres5.yx',
                   'path_validation': '../data_cvr3/cvr_imp_20180715_0721_conv_20180715_0727.csv.add_conv_type.thres5.yx',
                   'path_test': '../data_cvr3/cvr_imp_20180722_0728_conv_20180722_0803.csv.add_conv_type.thres5.yx',
                   'path_feature_index': '../data_cvr3/featureindex_thres5.txt',
                   'num_field': 17
                   }

with open('conf/project/cvr3.ini', 'w') as configfile:
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
