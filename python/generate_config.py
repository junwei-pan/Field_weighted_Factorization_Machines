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
