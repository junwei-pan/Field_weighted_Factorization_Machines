import sys
import json
import numpy as np

path = sys.argv[1]
n = 15
data = []

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
for i in range(n):
    l = []
    for j in range(n):
        l.append(d[i][j])
    lst.append(l)

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
