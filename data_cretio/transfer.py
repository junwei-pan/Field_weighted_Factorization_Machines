import sys

# Transfer the original cretio dataset to libsvm format
index_label = 0
lst_index_cat = range(14, 1 + 13 + 26)
num_field = 26
offset_train = 14
offset_test = 13
thres = 20

d_field_fea = {}
d_fea_index = {}
d_field_fea_cnt = {}
path_train = '/tmp/jwpan/data_cretio/train.txt'
path_fea_index = 'featindex_thres' + str(thres) + '.txt'
batch = 100000
total = 45840617

def build_field_feature(path, mode):
    for i, line in enumerate(open(path)):
        if i % batch == batch - 1:
            print i * 1.0 / total
            sys.stdout.flush()
        lst = line.strip('\n').split('\t')
        for idx_field in range(num_field):
            if mode == 'train':
                idx = idx_field + offset_train
            elif mode == 'test':
                idx = idx_field + offset_test
            fea = lst[idx]
            d_field_fea.setdefault(idx_field, set())
            d_field_fea[idx_field].add(fea)
            d_field_fea_cnt.setdefault(idx_field, {})
            d_field_fea_cnt[idx_field].setdefault(fea, 0) 
            d_field_fea_cnt[idx_field][fea] += 1

def create_fea_index(path):
    cnt_qualify = 0
    cnt_filter = 0
    index = 0
    file = open(path, 'w')
    for idx_field in range(num_field):
        for fea in d_field_fea[idx_field]:
            if d_field_fea_cnt[idx_field][fea] > thres:
                d_fea_index[fea] = index
                file.write("%d:%s\t%d\n" % (idx_field, fea, index))
                index += 1
                cnt_qualify += 1
            else:
                cnt_filter += 1
    print "number of features appears > %d times: %d" % (thres, cnt_qualify)
    print "number of features appears <= %d times: %d" % (thres, cnt_filter)
    file.close()

def create_yx(path, mode):
    # There is some samples whose all features are rare(# < thres), 
    # we need to filter all these samples, use cnt_filter as the counter.
    cnt_qualify = 0
    cnt_filter = 0
    file = open(path + '.thres' + str(thres) + '.yx', 'w')
    for i, line in enumerate(open(path)):
        if i % batch == batch - 1:
            print i * 1.0 / total
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
            if d_fea_index.has_key(fea):
                index = d_fea_index[fea]
                res.append("%d:1" % index)
            else:
                continue
        if len(res) > 1:
            cnt_qualify += 1
            file.write(' '.join(res) + '\n')
        else:
            cnt_filter += 1
    print "number of samples qualified: ", cnt_qualify
    print "number of samples with all rare features: ", cnt_filter
    file.close()

print 'build field feature'
build_field_feature(path_train, 'train')
#build_field_feature(path_test, 'test')

print 'create fea index'
create_fea_index(path_fea_index)

print 'create yx'
create_yx(path_train, 'train')
#create_yx(path_test, 'test')
