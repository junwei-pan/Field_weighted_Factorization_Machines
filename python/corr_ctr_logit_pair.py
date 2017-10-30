#! /usr/bin/env python

import pickle as pkl
import numpy as np
import itertools
from collections import defaultdict
import math
#import matplotlib.pyplot as plt
from scipy import stats

"""
    count most frequent features in each field, and choose top k
"""
def topK_feature1(data_list, zero_set, n_field, k):
    dic = defaultdict(int)
    for i in range(n_field):
        dic[i] = defaultdict(int)
    for tmp_list in data_list:
        for i in range(n_field):
            tmp_fea = tmp_list[i+1]
            if tmp_fea not in zero_set:
                dic[i][tmp_fea] += 1
    dic1 = defaultdict(int)
    for key, sub_dic in dic.items():
        sort_list = sorted(sub_dic.items(), key=lambda x:x[1])
        if len(sort_list) > k:
            sort_list = sort_list[-k:]
        dic1[key] = dict(sort_list)
    return dic1
        
def read_test_file(test_file):
    res = []
    with open(test_file) as infile:
        for line in infile:
            tmp_list = line.split(' ')
            tmp_res = []
            for s in tmp_list:
                tmp_res.append(int(s.split(':')[0]))
            res.append(tmp_res)
    infile.close()
    return res     

"""
    compute CTR for each feature pair
    loop through test file, keep records with feature1 and feature2
    record label, count for 1 and 0
    CTR = count(1) / (count(0) + count(1))
"""

def build_d_ctr(lst_data, n_field, thres=100):
    total_num_clicks = 0
    total_num_non_clicks = 0
    d = {}
    d_top = {}
    for lst in lst_data:
        label = lst[0]
        if label == 1:
            total_num_clicks += 1
        else:
            total_num_non_clicks += 1
        for i in range(n_field):
            for j in range(i+1, n_field):
                pair_field = str(i) + '_' + str(j)
                pair_feature = str(lst[i+1]) + '_' + str(lst[j+1])
                d.setdefault(pair_field, {})
                d[pair_field].setdefault(pair_feature, [0,0,0])
                d_top.setdefault(pair_field, set([]))
                if label == 1:
                    d[pair_field][pair_feature][0] += 1
                else:
                    d[pair_field][pair_feature][1] += 1
                d[pair_field][pair_feature][2] += 1
                if d[pair_field][pair_feature][2] >= thres:
                    d_top[pair_field].add(pair_feature)
    return (d, d_top, total_num_clicks, total_num_non_clicks)

def compute_ctr_fast(d, field1, field2, fea1, fea2, toal_num_clicks, total_num_non_clicks):
    pair_field = str(field1) + '_' + str(field2)
    pair_feature = str(fea1) + '_' + str(fea2)
    if not d.has_key(pair_field) or not d[pair_field].has_key(pair_feature):
        return -1, -1
    lst = d[pair_field][pair_feature]
    num_click = lst[0]
    num_non_click = lst[1]
    num_click_prime = toal_num_clicks - num_click
    num_non_click_prime = total_num_non_clicks - num_non_click
    return float(num_click) / (num_click + num_non_click), float(num_click_prime) / (num_click_prime + num_non_click_prime)
    
def compute_ctr(data_list, field1, field2, fea_ind1, fea_ind2):
    click_count1, no_count1, click_count2, no_count2 = 0, 0, 0, 0
    for tmp_list in data_list:
        label = tmp_list[0]
        if tmp_list[field1+1] != fea_ind1 or tmp_list[field2+1] != fea_ind2:
            if label == 1:
                click_count2 += 1
            else:
                no_count2 += 1

        else:
            if label == 1:
                click_count1 += 1
            else:
                no_count1 += 1

    if (click_count1 == 0 and no_count1 == 0) or (click_count2 == 0 and no_count2 == 0):
        return -1, float(click_count2) / (click_count2 + no_count2) 
    else:
        return float(click_count1) / (click_count1 + no_count1), float(click_count2) / (click_count2 + no_count2)

def find_index(n, f1, f2):
    ind = 0
    k = n - 1
    f = 0
    while f != f1:
        ind += k
        k -= 1
        f += 1
    return (ind + f2 - f1 - 1)     

def compute_dot(model_dic, common_text, filed1, field2, feature1, feature2, n_field):
    v1 = model_dic[common_text + str(field1)][feature1]
    v2 = model_dic[common_text + str(field2)][feature2]
    r12 = model_dic['w_p'][find_index(n_field, field1, field2)]
    dot = np.inner(v1, v2) * r12
    return dot[0]

#test_file = '/tmp/jwpan/data_yahoo/dataset2/ctr_20170531.txt.downsample_all.0.1.thres10.yx'
test_file = '../data_yahoo/dataset2/ctr_20170531.txt.downsample_all.0.1.thres10.yx'
model_file = 'model/fwfm_l2_v_1e-6_yahoo_dataset2.2_epoch_2'
feature_file = '../data_yahoo/dataset2/featindex_25m_thres10.txt'


### dictionary storing all fields, each field is a sub-dictionary storing top k feature ind
data_list = read_test_file(test_file)
model_dic = pkl.load(open(model_file, 'r'))

common_text = 'w0_'
num_field = 15
field_list = range(num_field)
### feature id is accumulated, this list stores the first index of each field
field_len_list = [0]
s = 0
for i in range(num_field-1):
    name = common_text + str(i)
    s += model_dic[name].shape[0]
    field_len_list.append(s)

print 'begin build_d_ctr()'
d, d_top, total_num_clicks, total_num_non_clicks = build_d_ctr(data_list, num_field, 1000)
print 'end build_d_ctr()'
small_float = 0.0000001

for pair in itertools.combinations(field_list, 2):
    field1, field2 = pair[0], pair[1]
    lst_ctr, lst_inner = [], []
    pair_field = str(field1) + '_' + str(field2)
    for pair_feature in d_top[pair_field]:
        fea_ind1, fea_ind2 = map(int, pair_feature.split('_'))
        feature1 = fea_ind1 - field_len_list[field1]
        feature2 = fea_ind2 - field_len_list[field2]
        ctr1, ctr2 = compute_ctr_fast(d, field1, field2, fea_ind1, fea_ind2, total_num_clicks, total_num_non_clicks)
        if ctr1 < 0:
            continue
        inner = compute_dot(model_dic, common_text, field1, field2, feature1, feature2, num_field)
        lst_ctr.append(math.log(ctr1 / (1 - ctr1 + small_float) + small_float) - math.log(ctr2 / (1 - ctr2 + small_float) + small_float))
        lst_inner.append(inner)

    out = open('data/corr_results/result_field'+str(field1)+'_field'+str(field2)+'.txt', 'w+')
    res = stats.pearsonr(lst_ctr, lst_inner)
    res = list(res) + [len(d_top[pair_field])]
    print ','.join(map(str, res))
    out.write(','.join(map(str, lst_ctr)) + '\n')
    out.write(','.join(map(str, lst_inner)) + '\n')
    out.close()

