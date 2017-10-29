import sys
import mmh3

'''
This script will do the following tasks:
    1. Generate a feature index file which maps each feature to an unique index.
    2. Transfer the original tsv files to libsvm format.
'''

# Configuration for Criteo data set
index_label = 0
index_cat_start = 14
num_fields = 26
lst_index_cat = range(index_cat_start, index_cat_start + num_fields)
thres = 20

# Configuration for Yahoo data set
'''
index_label = 28
num_fields = 15
lst_index_cat = range(num_fields)
thres = 10
'''

print "Index of label", index_label
print "List of indexes of categorical features", lst_index_cat
print "Number of fields", num_fields
print "Threshold", thres

d_field_fea = {}
d_fea_index = {}
d_field_fea_cnt = {}
# Yahoo CTR data set.
'''
path_train = '../data_yahoo/ctr_20170517_0530_0.015.txt'
path_validation = '../data_yahoo/ctr_20170531.txt.downsample_all.0.1'
path_test = '../data_yahoo/ctr_20170531.txt.downsample_all.0.1'
'''

# Criteo CTR data set.
path_train = '/tmp/jwpan/data_cretio/train.txt.train.100k'
path_validation = '/tmp/jwpan/data_cretio/train.txt.validation.100k'
path_test = '/tmp/jwpan/data_cretio/train.txt.test.100k'
path_fea_index = '/tmp/jwpan/data_cretio/featindex_thres20.txt'

batch = 100000

def get_lines_of_file(path):
    n = 0
    for i in open(path):
        n += 1
    return n

def build_field_feature(path, mode):
    '''
    Output:
        d_field_fea, key: field index, value: set of features of a field
        d_field_fea_cnt, key: field index, value: {key: feature index, value: count of samples with this feature}
    '''
    total = get_lines_of_file(path)
    for i, line in enumerate(open(path)):
        if i % batch == batch - 1:
            print i * 1.0 / total
            sys.stdout.flush()
        lst = line.strip('\n').split('\t')
        for idx_field in range(num_fields):
            fea = lst[index_cat_start + idx_field]
            d_field_fea.setdefault(idx_field, set())
            d_field_fea[idx_field].add(fea)
            d_field_fea_cnt.setdefault(idx_field, {})
            d_field_fea_cnt[idx_field].setdefault(fea, 0) 
            d_field_fea_cnt[idx_field][fea] += 1

def create_fea_index(path, model):
    cnt_qualify = 0
    cnt_filter = 0
    # The feature index for LibLinear package should start from 1 rather than from 0.
    if model == 'liblinear':
        index = 1
        file = open(path + '.liblinear', 'w')
    else:
        index = 0
        file = open(path, 'w')
    for idx_field in range(num_fields):
        d_fea_index.setdefault(idx_field, {})
        d_fea_index[idx_field]['zero_fea_for_field_' + str(idx_field)] = index
        file.write("%d:%s\t%d\n" % (idx_field, 'zero_fea_for_field_' + str(idx_field), index))
        index += 1
        for fea in d_field_fea[idx_field]:
            if d_field_fea_cnt[idx_field][fea] > thres:
                d_fea_index[idx_field][fea] = index
                file.write("%d:%s\t%d\n" % (idx_field, fea, index))
                index += 1
                cnt_qualify += 1
            else:
                cnt_filter += 1
    print "number of features appears > %d times: %d" % (thres, cnt_qualify)
    print "number of features appears <= %d times: %d" % (thres, cnt_filter)
    file.close()

def create_ffm_fea_index(path, d=14, k=2):
    '''
    In order to compare FFM with FwFM using the same size of parameters and hence memory, we need to use
    hashing trick with a small hashing space.
    '''
    d = 14.0 / (10.0 / k)
    cnt_fea_before_hash = 0
    cnt_fea_after_hash = 0
    index = 0
    file = open(path, 'w')
    for idx_field in range(num_fields):
        cnt_qualify_for_this_field = 0
        d_fea_index.setdefault(idx_field, {})
        d_fea_index[idx_field]['zero_fea_for_field_' + str(idx_field)] = index
        file.write("%d:%s\t%d\n" % (idx_field, 'zero_fea_for_field_' + str(idx_field), index))
        cnt_fea_after_hash += 1
        index += 1
        for fea in d_field_fea[idx_field]:
            if d_field_fea_cnt[idx_field][fea] > thres:
                cnt_fea_before_hash += 1
                cnt_qualify_for_this_field += 1
        print "idx_field: %d, cnt_qualify_for_this_field: %d" % (idx_field, cnt_qualify_for_this_field)
        if cnt_qualify_for_this_field >= 10 * d:
            start = index
            end = index + cnt_qualify_for_this_field / d
            for fea in d_field_fea[idx_field]:
                if d_field_fea_cnt[idx_field][fea] > thres:
                    idx_hash = mmh3.hash(fea) % (end - start) + start
                    d_fea_index[idx_field][fea] = idx_hash
                    file.write("%d:%s\t%d\n" % (idx_field, fea, idx_hash))
            index += end - start
            cnt_fea_after_hash += end - start
            print "field %d qualifies, # parameters before hash: %d, # parameters after hash: %d" % (idx_field, cnt_qualify_for_this_field, end - start)
        else:
            for fea in d_field_fea[idx_field]:
                if d_field_fea_cnt[idx_field][fea] > thres:
                    cnt_fea_after_hash += 1
                    d_fea_index[idx_field][fea] = index
                    file.write("%d:%s\t%d\n" % (idx_field, fea, index))
                    index += 1
            #else:
    #print "number of features appears > %d times: %d" % (thres, cnt_qualify)
    #print "number of features appears <= %d times: %d" % (thres, cnt_filter)
    print "cnt_fea_before_hash: %d" % cnt_fea_before_hash
    print "cnt_fea_after_hash: %d" % cnt_fea_after_hash
    file.close()

def create_yx(path, mode, model='fm', d=14, k=2):
    '''
    There is some samples whose all features are rare(# < thres), 
    we need to filter all these samples, use cnt_filter as the counter.
    '''
    cnt_qualify = 0
    cnt_filter = 0
    total = get_lines_of_file(path)
    if model == 'ffm':
        suffix = str(round(d / (10.0 / k), 2))
        file = open(path + '.thres' + str(thres) + '.ffm' + suffix + '.yx', 'w')
    elif model == 'liblinear':
        file = open(path + '.thres' + str(thres) + '.liblinear.yx', 'w')
    else:
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
        for idx in range(num_fields):
            fea = lst[index_cat_start + idx]
            if i == 0:
                print 'idx: %d, fea: %s' % (idx, fea)
            if d_fea_index.has_key(idx) and d_fea_index[idx].has_key(fea):
                index = d_fea_index[idx][fea]
                res.append("%d:1" % index)
            else:
                index = d_fea_index[idx]['zero_fea_for_field_' + str(idx)]
                res.append("%d:1" % index)
        if len(res) > 1:
            cnt_qualify += 1
            file.write(' '.join(res) + '\n')
        else:
            cnt_filter += 1
    print "number of samples qualified: ", cnt_qualify
    print "number of samples with all rare features: ", cnt_filter
    file.close()

# To generate several data sets for FFM with different hashing space.
d = 26 - 1

print 'build field feature'
build_field_feature(path_train, 'train')

for k in range(1, 10):
    suffix = str(round(d / (10.0 / k), 2))
    path_fea_ffm_index = '/tmp/jwpan/data_cretio/featindex_ffm'+ suffix +'_thres' + str(thres) + '.txt'

    print 'create fea index'
    #create_fea_index(path_fea_index)
    create_ffm_fea_index(path_fea_ffm_index, d, k)

    print 'create yx'
    create_yx(path_train, 'train', 'ffm', d, k)
    create_yx(path_validation, 'train', 'ffm', d, k)
    create_yx(path_test, 'train', 'ffm', d, k)

'''
flagLibLinear = False
model = 'liblinear'
print ' ===> model: %s <=== ' % model

print 'build field feature'
build_field_feature(path_train, 'train')

print 'create fea index'
create_fea_index(path_fea_index, model)
#create_ffm_fea_index(path_fea_ffm_index, d, k)

print 'create yx'
create_yx(path_train, 'train', model)
create_yx(path_validation, 'train', model)
create_yx(path_test, 'train', model)
'''
