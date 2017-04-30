# Transfer the original cretio dataset to libsvm format
index_label = 0
lst_index_cat = range(14, 1 + 13 + 26)
num_field = 26
offset_train = 14
offset_test = 13

d_field_fea = {}
d_fea_index = {}
path_train = 'train.txt'
path_fea_index = 'featindex.txt'

def build_field_feature(path, mode):
    for i, line in enumerate(open(path)):
        lst = line.strip('\n').split('\t')
        for idx_field in range(num_field):
            if mode == 'train':
                idx = idx_field + offset_train
            elif mode == 'test':
                idx = idx_field + offset_test
            fea = lst[idx]
            d_field_fea.setdefault(idx_field, set())
            d_field_fea[idx_field].add(fea)

def create_fea_index(path):
    index = 0
    file = open(path, 'w')
    for idx_field in range(num_field):
        for fea in d_field_fea[idx_field]:
            d_fea_index[fea] = index
            file.write("%d:%s\t%d\n" % (idx_field, fea, index))
            index += 1
    file.close()

def create_yx(path, mode):
    file = open(path + '.yx', 'w')
    for i, line in enumerate(open(path)):
        res = []
        lst = line.strip('\n').split('\t')
        if mode == 'train':
            res.append(lst[index_label])
        elif mode == 'test':
            res.append('0')
        for idx_field in range(num_field):
            if mode == 'train':
                idx = idx_field + offset_train
            elif mode == 'test':
                idx = idx_field + offset_test
            fea = lst[idx]
            index = d_fea_index[fea]
            res.append("%d:1" % index)
        file.write(' '.join(res) + '\n')
    file.close()

build_field_feature(path_train, 'train')
#build_field_feature(path_test, 'test')

create_fea_index(path_fea_index)

create_yx(path_train, 'train')
#create_yx(path_test, 'test')
