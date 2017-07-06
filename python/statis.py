import pickle as pkl
import numpy

class statis:
    def __init__(self):
        self.k = 15
        self.d = {}
        self.d_idx2embedding = {}
        self.d_idx2idxField = {}

    def load_model(self, path_model, path_fea_idx):
        self.d = pkl.load(open(path, 'rb'))
        self.idx2key = {}
        idx_last_field = 0
        name_last_field = '0'
        total_idx = 0
        for i in range(self.k):
            d_field_i = self.d['w0_' + str(i)]
            for j in range(len(d_field_i)):
                self.idx2embedding[total_idx] = d_field_i[j]
                total_idx += 1
            
        for line in open(path_fea_idx):
            lst = line.strip('\n').split('\t')
            idx_field = int(lst[0].split(':')[0])
            idx = int(lst[1])
            self.d_idx2idxField[idx] = idx_field


    def get_feature_dot_product(self, i, j):
        return 

            
            

    def get_embedding(self, idx):
        '''
        Return the embedding vector for a feature with index idx
        '''
        return 
    
