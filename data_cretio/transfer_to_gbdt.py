
lst_index_cat = range(14, 1 + 13 + 26)
path = 'train.txt.10000000'
file = open(path + '.fv', 'w')
for idx, line in enumerate(open(path)):
    lst = line.strip("\n").split("\t")
    if idx == 0:
        file.write('\t'.join(["cat_" + str(i) + '$' for i in range(26)] + ['label']) + '\n')
    res = []
    for index in lst_index_cat:
        res.append(lst[index])
    res.append(lst[0])
    file.write('\t'.join(res) + '\n')
file.close()
