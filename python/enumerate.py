import sys
import json
import numpy as np

path = sys.argv[1]
n = 17
data = []
lst_field = ['PUBLISHER', 'PAGE_TLD',    'SUBDOMAIN',   'LAYOUT',   'DAY_OF_WEEK', 'HOUR_OF_DAY', 'GENDER', 'AD_PLACEMENT',
     'AD_POSITION', 'AGE_BUCKET', 'ADVERTISER', 'AD', 'CRRATIVE', 'CREATIVE_MEDIA', 'DEVICE_TYPE',
     'LINE', 'USER']
for line in open(path):
    data.append(line.strip('\n'))
d_idx_pair = {}
idx = 0
for i in range(n):
    for j in range(i+1, n):
        d_idx_pair[idx] = str(i) + '_' + str(j)
        idx += 1

d = {}
for idx, e in enumerate(data):
    i, j = map(int, d_idx_pair[idx].split('_'))
    d.setdefault(i, {})
    d.setdefault(j, {})
    d[i][j] = e
    d[j][i] = e

for i in range(n):
    d[i][i] = 0

#print d

for i in range(n):
    res = []
    for j in range(n):
        res.append(d[i][j])
    print '\t'.join(map(str, res))

lst = []
lst_with_index = []
for i in range(n):
    l = []
    for j in range(n):
        l.append(d[i][j])
        lst_with_index.append([float(d[i][j]), i, j])
    lst.append(l)

# View Content: 1, Purchase: 2, Sign Up: 3, Lead: 4
sort = sorted(lst_with_index, key=lambda x:x[0], reverse=True)
for i in range(10):
    print sort[i][0], lst_field[sort[i][1]], lst_field[sort[i][2]]

with open(sys.argv[1] + '.json', 'w') as out:
    json.dump(lst, out)


'''
for i in range(n):
    res = []
    for j in range(i+1, n):
        idx += 1
        res.append(data[idx])
    print '\t'.join(res)
'''
