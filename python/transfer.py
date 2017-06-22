import sys

# Transfer the original cretio dataset to libsvm format
index_label = 28
#lst_index_cat = range(14, 14 + 26)
lst_index_cat = range(15)
#num_field = 26
#offset_train = 14
#offset_test = 13
thres = 10

print "index_label", index_label
print "lst_index_cat", lst_index_cat

d_field_fea = {}
d_fea_index = {}
d_field_fea_cnt = {}
#path_train = '/tmp/jwpan/data_cretio/train.txt'
#path_train = '../data_yahoo/ctr_20170524_0530_0.003.txt'
path_train = '../data_yahoo/ctr_20170517_0530_0.015.txt'
path_test = '../data_yahoo/ctr_20170531.txt.downsample_all.0.1'
#path_fea_index = '../data_yahoo/featindex_3m_thres' + str(thres) + '.txt'
path_fea_index = '../data_yahoo/featindex_25m_thres' + str(thres) + '.txt'
batch = 100000

def get_lines_of_file(path):
    n = 0
    for i in open(path):
        n += 1
    return n

def build_field_feature(path, mode):
    total = get_lines_of_file(path)
    for i, line in enumerate(open(path)):
        if i % batch == batch - 1:
            print i * 1.0 / total
            sys.stdout.flush()
        lst = line.strip('\n').split('\t')
        for idx_field in lst_index_cat:
            fea = lst[idx_field]
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
    for idx_field in lst_index_cat:
        for fea in d_field_fea[idx_field]:
            if d_field_fea_cnt[idx_field][fea] > thres:
                d_fea_index.setdefault(idx_field, {})
                d_fea_index[idx_field][fea] = index
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
    total = get_lines_of_file(path)
    file = open(path + '.thres' + str(thres) + '.yx', 'w')
    for i, line in enumerate(open(path)):
        if i % batch == batch - 1:
            print i * 1.0 / total
        res = []
        lst = line.strip('\n').split('\t')
        if mode == 'train':
            label = lst[index_label]
            if label == '-1':
                label = '0'
            res.append(label)
            if i == 0:
                print 'label', label
        elif mode == 'test':
            res.append('0')
        for idx in lst_index_cat:
            fea = lst[idx]
            if i == 0:
                print 'idx: %d, fea: %s' % (idx, fea)
            if d_fea_index.has_key(idx) and d_fea_index[idx].has_key(fea):
                index = d_fea_index[idx][fea]
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

print 'create fea index'
create_fea_index(path_fea_index)

print 'create yx'
create_yx(path_train, 'train')
create_yx(path_test, 'train')
#create_yx(path_test, 'test')
