import pickle as pkl
import numpy as np
import scipy
from scipy.stats.stats import pearsonr

class statis:
    def __init__(self):
        self.k = 15
        self.d = {}
        self.d_idx2embedding = {}
        self.d_idx2idxField = {}
        self.lst_fea = []
        self.lst_label = []

    def load_model(self, path_model, path_fea_idx):
        self.d = pkl.load(open(path_model, 'rb'))
        self.idx2key = {}
        idx_last_field = 0
        name_last_field = '0'
        total_idx = 0
        for i in range(self.k):
            d_field_i = self.d['w0_' + str(i)]
            for j in range(len(d_field_i)):
                self.d_idx2embedding[total_idx] = d_field_i[j]
                total_idx += 1
            
        for line in open(path_fea_idx):
            lst = line.strip('\n').split('\t')
            idx_field = int(lst[0].split(':')[0])
            idx = int(lst[1])
            self.d_idx2idxField[idx] = idx_field

    def load_data(self, path):
        for line in open(path):
            lst = line.strip('\n').split(' ')
            label = int(lst[0])
            lst_fea = [int(x.split(':')[0]) for x in lst[1:]]
            self.lst_fea.append(lst_fea)
            self.lst_label.append(label)

    def get_feature_dot_product(self, i, j):
        return np.dot(self.d_idx2embedding[i], self.d_idx2embedding[j])

    def get_field_corr(self, fi, fj):
        '''
        This will get a list, whose value is <v_i, v_j>, where i, j belongs to fi and fj respectively.
        '''
        res = []
        for lst_fea in self.lst_fea:
            i = lst_fea[fi]
            j = lst_fea[fj]
            res.append(self.get_feature_dot_product(i, j))
        return res

    def get_field_pair_pearson_corr_with_label(self, fi, fj):
        lst_score = self.get_field_corr(fi, fj)
        return pearsonr(lst_score, self.lst_label)
            
    def get_embedding(self, idx):
        '''
        Return the embedding vector for a feature with index idx
        '''
        pass

statis = statis()
statis.load_data('../data_yahoo/ctr_20170531.txt.downsample_all.0.1.thres10.yx')
statis.load_model('model/yahoo_dataset2.2_fwfm_0.0005_epoch_2', '../data_yahoo/featindex_25m_thres10.txt')
for i in range(15):
    for j in range(i+1, 15):
        res = statis.get_field_pair_pearson_corr_with_label(i,j)
        print '%d\t%d\t%f\t%f' % (i, j, res[0], res[1])
