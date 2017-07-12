import sys

path = sys.argv[1]
print 'path', path

file = open(path + '.libffm', 'w')

for line in open(path):
    lst = line.strip('\n').split(' ')
    res = []
    label = lst[0]
    res.append(label)
    for i in range(len(lst[1:])):
        idx = lst[1 + i].split(':')[0]
        res.append(str(i) + ':' + idx + ':1')
    file.write(' '.join(res) + '\n')
file.close()
