import sys
import pickle as pkl
import numpy as np
import scipy
import math
from scipy.stats.stats import pearsonr
from scipy.stats import kendalltau

class statis:
    def __init__(self, M=0):
        self.M = M # Number of fields.
        self.k = 10 # Dimension of embedding vector.
        self.n_task = -1
        self.d = {}
        self.d_idx2embedding = {}
        self.d_idx2idxField = {}
        self.lst_fea = []
        self.set_fea = set([])
        self.lst_label = []
        self.d_indexField_feature = {} # key: fi, value: {feature: frequency}
        self.d_fieldPair_featurePair = {} # key: fi_fj, value: {fea_i_fea_j: frequency}
        self.cnt_sample = 0.0
        self.cnt_pos = 0.0
        self.cnt_neg = 0.0
        self.d_fieldPair_r = {}
        self.d_conv_type_fieldPair_r = {}

    def load_model(self, path_model, model = 'fwfm'):
        self.d = pkl.load(open(path_model, 'rb'))
        self.idx2key = {}
        idx_last_field = 0
        name_last_field = '0'
        total_idx = 0
        if model == 'fwfm' or model == 'mtl-fwfm' or model == 'mtl-fwfm-r-factorized':
            for i in range(self.M):
                d_field_i = self.d['w0_' + str(i)]
                for j in range(len(d_field_i)):
                    self.d_idx2embedding[total_idx] = d_field_i[j]
                    total_idx += 1
            if model == 'fwfm':
                idx = 0
                for i in range(self.M):
                    for j in range(i+1, self.M):
                        field_pair = str(i) + '_' + str(j)
                        self.d_fieldPair_r[field_pair] = self.d['w_p'][idx][0]
                        idx += 1
            elif model == 'mtl-fwfm':
                self.n_task = len(self.d['r'])
                for idx_task in range(len(self.d['r'])):
                    idx = 0
                    for i in range(self.M):
                        for j in range(i+1, self.M):
                            field_pair = str(i) + '_' + str(j)
                            self.d_conv_type_fieldPair_r.setdefault(idx_task, {})
                            self.d_conv_type_fieldPair_r[idx_task][field_pair] = self.d['r'][idx_task][idx]
                            idx += 1
            elif model == 'mtl-fwfm-r-factorized':
                self.n_task = len(self.d['r_factorized'])
                for idx_task in range(self.n_task):
                    for i in range(self.M):
                        for j in range(i+1, self.M):
                            field_pair = str(i) + '_' + str(j)
                            self.d_conv_type_fieldPair_r.setdefault(idx_task, {})
                            self.d_conv_type_fieldPair_r[idx_task][field_pair] = np.dot(self.d['r_factorized'][idx_task][i], self.d['r_factorized'][idx_task][j])
        elif model == 'ffm':
            for i in range(self.M):
                d_field_i = self.d['w0_' + str(i)]
                for j in range(len(d_field_i)):
                    self.d_idx2embedding[total_idx] = []
                    for l in range(self.M):
                        self.d_idx2embedding[total_idx].append(d_field_i[j][self.k * l : self.k * (l + 1)])
                    total_idx += 1
            print self.d_idx2embedding.keys()
        elif model == 'fm':
            for i in range(len(self.d['v'])):
                self.d_idx2embedding[i] = self.d['v'][i]
            
    def load_feature_index(self, path):
        for line in open(path):
            lst = line.strip('\n').split('\t')
            idx_field = int(lst[0].split(':')[0])
            idx = int(lst[1])
            self.d_idx2idxField[idx] = idx_field

    def load_data(self, path):
        cnt_pos = 0
        cnt_neg = 0
        bin = 100000
        for idx_line, line in enumerate(open(path)):
            self.cnt_sample += 1
            if idx_line % bin == bin - 1:
                print idx_line
                sys.stdout.flush()
            lst = line.strip('\n').split(' ')
            label = int(lst[0])
            if label == 1:
                self.cnt_pos += 1
            else:
                self.cnt_neg += 1
            lst_fea = [int(x.split(':')[0]) for x in lst[1:]]
            for i in range(len(lst_fea)):
                fea = lst_fea[i]
                self.set_fea.add(fea)
                self.d_indexField_feature.setdefault(i, {})
                self.d_indexField_feature[i].setdefault(fea, 0)
                self.d_indexField_feature[i][fea] += 1
                for j in range(i+1, len(lst_fea)):
                    fea_j = lst_fea[j]
                    field_pair = str(i) + '_' + str(j)
                    feature_pair = str(fea) + '_' + str(fea_j)
                    self.d_fieldPair_featurePair.setdefault(field_pair, {})
                    self.d_fieldPair_featurePair[field_pair].setdefault(feature_pair, {'cnt':0.01, 'pos':0.01, 'neg':0.01})
                    self.d_fieldPair_featurePair[field_pair][feature_pair]['cnt'] += 1
                    if label == 1:
                        self.d_fieldPair_featurePair[field_pair][feature_pair]['pos'] += 1
                        cnt_pos += 1
                    else:
                        self.d_fieldPair_featurePair[field_pair][feature_pair]['neg'] += 1
                        cnt_neg += 1
            self.lst_fea.append(lst_fea)
            self.lst_label.append(label)
        print 'len(self.set_fea)', len(self.set_fea)
        print 'cnt_pos: %d, cnt_neg: %d, cvr: %f' % (cnt_pos, cnt_neg, cnt_pos * 1.0 / (cnt_pos + cnt_neg))

    def basic_statistics(self, path_data, d_index_field={}):
        print 'd_index_field', d_index_field
        print path_data
        cnt_pos = 0
        cnt_neg = 0
        for line in open(path_data):
            lst = line.strip('\n').split(' ')
            label = int(lst[0])
            lst_fea = [int(x.split(':')[0]) for x in lst[1:]]
            for i in range(len(lst_fea)):
                fea = lst_fea[i]
                self.set_fea.add(fea)
                self.d_indexField_feature.setdefault(i, {})
                self.d_indexField_feature[i].setdefault(fea, 0)
            if label == 1:
                cnt_pos += 1
            else:
                cnt_neg += 1
        print 'len(self.set_fea)', len(self.set_fea)
        for i in range(self.M):
            print i, d_index_field[i], len(self.d_indexField_feature[i].keys())
        print 'cnt_total: %d, cnt_pos: %d, cnt_neg: %d, cvr: %f' % (cnt_pos + cnt_neg, cnt_pos, cnt_neg, cnt_pos * 1.0 / (cnt_pos + cnt_neg))

    def get_feature_dot_product(self, i, j, fi, fj, model = 'fm'):
        if model == 'ffm':
            return np.dot(self.d_idx2embedding[i][fj], self.d_idx2embedding[j][fi])
        else:
            return np.dot(self.d_idx2embedding[i], self.d_idx2embedding[j])

    def get_field_corr(self, fi, fj):
        '''
        This will get a list, whose value is <v_i, v_j>, where i, j belongs to fi and fj respectively.
        '''
        res = []
        for lst_fea in self.lst_fea:
            i = lst_fea[fi]
            j = lst_fea[fj]
            res.append(self.get_feature_dot_product(i, j, fi, fj, model))
        return res

    def get_field_pair_pearson_corr_with_label(self, fi, fj):
        lst_score = self.get_field_corr(fi, fj)
        return pearsonr(lst_score, self.lst_label)
            
    def get_embedding(self, idx):
        '''
        Return the embedding vector for a feature with index idx
        '''
        pass

    def average_latent_vector_dot_product_for_field_pair(self, fi, fj, model ='fwfm', n_task=0):
        sum = 0.0
        sum_abs = 0.0
        sum_cnt = 0.0
        uniq_fea_pair_cnt = 0.0
        field_pair = str(fi) + '_' + str(fj)
        if model == 'fwfm':
            for feature_pair in self.d_fieldPair_featurePair[field_pair]:
                cnt = self.d_fieldPair_featurePair[field_pair][feature_pair]['cnt']
                r = self.d_fieldPair_r[field_pair]
                fea_i, fea_j = map(int, feature_pair.split('_'))
                dot = self.get_feature_dot_product(fea_i, fea_j, fi, fj, model)
                sum += dot * cnt * r
                sum_abs += abs(dot * r) * cnt
                sum_cnt += cnt
                uniq_fea_pair_cnt += 1
            return sum, sum_abs, sum_cnt, uniq_fea_pair_cnt
        elif model == 'mtl-fwfm' or model == 'mtl-fwfm-r-factorized':
            for feature_pair in self.d_fieldPair_featurePair[field_pair]:
                cnt = self.d_fieldPair_featurePair[field_pair][feature_pair]['cnt']
                r = self.d_conv_type_fieldPair_r[n_task][field_pair]
                fea_i, fea_j = map(int, feature_pair.split('_'))
                dot = self.get_feature_dot_product(fea_i, fea_j, fi, fj, model)
                sum += dot * cnt * r
                sum_abs += abs(dot * r) * cnt
                sum_cnt += cnt
                uniq_fea_pair_cnt += 1
            return sum, sum_abs, sum_cnt, uniq_fea_pair_cnt, abs(r)

            
    def mutual_information(self, fi, fj):
        mi = 0.0
        field_pair = str(fi) + '_' + str(fj)
        for feature_pair in self.d_fieldPair_featurePair[field_pair]:
            p_fi_fj = self.d_fieldPair_featurePair[field_pair][feature_pair]['cnt'] / self.cnt_sample
            p_fi_fi_pos = self.d_fieldPair_featurePair[field_pair][feature_pair]['pos'] / self.cnt_sample
            p_fi_fi_neg = self.d_fieldPair_featurePair[field_pair][feature_pair]['neg'] / self.cnt_sample
            p_pos = self.cnt_pos / self.cnt_sample
            p_neg = self.cnt_neg / self.cnt_sample
            mi += p_fi_fi_pos * math.log(p_fi_fi_pos / (p_fi_fj * p_pos))
            mi += p_fi_fi_neg * math.log(p_fi_fi_neg / (p_fi_fj * p_neg))
        return mi
        
def load_list(path):
    lst = []
    for line in open(path):
        lst.append(float(line.strip('\n')))
    return lst

def statis_n_feature(path):
    set_fea = set([])
    for line in open(path):
        lst = [x.split(':')[0] for x in line.strip('\n').split(' ')[1:]]
        for x in lst:
            set_fea.add(x)
    print len(set_fea)
'''
path1 = '../data_yahoo/dataset2/ctr_20170517_0530_0.015.txt.thres10.yx'
statis_n_feature(path1)
path2 = '../data_yahoo/dataset2/ctr_20170531.txt.downsample_all.0.1.thres10.yx'
statis_n_feature(path2)
path3 = '../data_yahoo/dataset2/ctr_20170601.txt.downsample_all.0.1.thres10.yx'
statis_n_feature(path3)
path1 = '../data_cretio/train.txt.train.thres20.yx'
statis_n_feature(path1)
path2 = '../data_cretio/train.txt.validation.thres20.yx'
statis_n_feature(path2)
path3 = '../data_cretio/train.txt.test.thres20.yx'
statis_n_feature(path3)
'''
# View Content: 1, Purchase: 2, Sign Up: 3, Lead: 4
d_index_task = {1: 'View_Content', 2: 'Purchase', 3: 'Sign_Up', 4: 'Lead'}
d_index_field = {0: 'publisher_id',
                 1: 'page_tld',
                 2: 'subdomain',
                 3: 'layout_id',
                 4: 'user_local_day_of_week',
                 5: 'user_local_hour',
                 6: 'gender',
                 7: 'ad_placement_id',
                 8: 'ad_position_id',
                 9: 'age',
                 10: 'account_id',
                 11: 'ad_id',
                 12: 'creative_id',
                 13: 'creative_media_id',
                 14: 'device_type_id',
                 15: 'line_id',
                 16: 'user_id'}
num_task = len(d_index_task.keys())
index_task = 3
n_field = 17
#path_model = 'model/MTLfwfm_lr_5e-5_l2_v_1e-5_yahoo_dataset2.2_epoch_72'
#path_model = 'model/MTLfwfm_r_factorized_lr_5e-5_yahoo_dataset2.2_epoch_50'
#path_model = 'model/MTLfwfm_r_factorized_lr_5e-5_l2_r_1e-5_yahoo_dataset2.2_epoch_46'
path_model = 'model/MTLfwfm_lr_5e-5_l2_v_1e-5_yahoo_dataset2.2_epoch_83'
#path_data = '../data_cvr/cvr_imp_20180704_0710_conv_20180704_0716.csv.add_conv_type.thres5.yx.' + d_index_task[index_task]
path_data = '../data_cvr/cvr_imp_20180711_conv_20180711_0717.csv.add_conv_type.thres5.yx.' + d_index_task[index_task]

def statis_mtl_fwfm_r(path_data, path_model, model='fwfm'):
    # Possible argument for model: fwfm, mtl-fwfm, mtl-fwfm-r-factorized
    s = statis(M=n_field)
    print 'load data: %s' % path_data
    s.load_data(path_data)
    print 'load model: %s' % path_model
    s.load_model(path_model, model)
    file = open(path_model + '.task_' + str(index_task), 'w')
    file_r = open(path_model + '.task_' + str(index_task) + '.r', 'w')
    for fi in range(n_field):
        for fj in range(fi+1, n_field):
            res = s.average_latent_vector_dot_product_for_field_pair(fi, fj, model, index_task)
            print 'fi: %d, fj: %d, sum: %f, sum_abs: %f, sum_cnt: %s, uniq_fea_pair_cnt: %f' % (fi, fj, res[0], res[1], res[2], res[3])
            file.write(str(res[1] / res[2]) + '\n')
            file_r.write(str(res[4]) + '\n')
    file.close()
    file_r.close()

def statis_feature_and_label_for_dataset(path_data, d_index_field={}):
    s = statis(M=n_field)
    s.basic_statistics(path_data, d_index_field)

def statis_pearson_correlation():
    d_conv_type_mi = {}
    d_conv_type_r = {}
    for index_task in d_index_task.keys():
        conv_type = d_index_task[index_task]
        print conv_type
        d_conv_type_mi[conv_type] = load_list('data/mi_cvr_' + conv_type.lower())
        #d_conv_type_r[conv_type] = load_list('model/MTLfwfm_lr_5e-5_l2_v_1e-5_yahoo_dataset2.2_epoch_72.task_' + str(index_task) + '.r')
        d_conv_type_r[conv_type] = load_list('model/MTLfwfm_r_factorized_lr_5e-5_yahoo_dataset2.2_epoch_50.task_' + str(index_task) + '.r')
    for i in range(num_task):
        for j in range(num_task):
            index_i = i + 1
            index_j = j + 1
            conv_type_i = d_index_task[index_i]
            conv_type_j = d_index_task[index_j]
            print conv_type_i, conv_type_j, round(pearsonr(d_conv_type_mi[conv_type_i], d_conv_type_r[conv_type_j])[0], 2)

#statis_mtl_fwfm_r(path_data, path_model, model='mtl-fwfm')
print 'd_index_field', d_index_field
statis_feature_and_label_for_dataset('../data_cvr/cvr_imp_20180711_conv_20180711_0717.csv.add_conv_type.thres5.yx.Lead',d_index_field=d_index_field)
#statis_pearson_correlation()

'''
#print 'load feature index'
#sys.stdout.flush()
#statis.load_feature_index('../data_yahoo/dataset2/featindex_25m_thres10.txt')
sys.stdout.flush()
#statis.load_data('../data_yahoo/dataset2/ctr_20170517_0530_0.015.txt.thres10.yx')
sys.stdout.flush()
#statis.load_model('model/yahoo_dataset2.2_fwfm_epoch_2', 'fwfm')
#statis.load_model('model/ffm_l2_v_1e-7_lr_1e-4_yahoo_epoch_2', 'ffm')
for fi in range(17):
    for fj in range(fi+1, 17):
        #res = statis.average_latent_vector_dot_product_for_field_pair(fi, fj, 'fwfm')
        #print "%f\t%f\t%f\t%f" % (res[0], res[1], res[2], res[3])
        #sys.stdout.flush()
        res = statis.mutual_information(fi, fj)
        print res
        #res = statis.get_field_pair_pearson_corr_with_label(i,j)
        #print '%d\t%d\t%f\t%f' % (i, j, res[0], res[1])
'''

def main_kendalltau():
    path_mi = 'data/yahoo_mi'
    path_fm = 'data/yahoo_fm'
    path_ffm = 'data/yahoo_ffm'
    path_fwfm = 'data/yahoo_fwfm'
    path_r = 'data/yahoo_abs_r'
    path_fwfm_without_r = 'data/yahoo_fwfm_without_r'
    x_mi = load_list(path_mi)
    x_fm = load_list(path_fm)
    x_ffm = load_list(path_ffm)
    x_fwfm = load_list(path_fwfm)
    x_r = load_list(path_r)
    x_fwfm_without_r = load_list(path_fwfm_without_r)
    print 'mi v.s fm'
    print pearsonr(x_mi, x_fm)
    print 'mi v.s ffm'
    print pearsonr(x_mi, x_ffm)
    print 'mi v.s fwfm'
    print pearsonr(x_mi, x_fwfm)
    print 'mi v.s r'
    print pearsonr(x_mi, x_r)
    print 'mi v.s fwfm_without_r'
    print pearsonr(x_mi, x_fwfm_without_r)
