import random

path = '/tmp/jwpan/data_cretio/train.txt.thres20.yx'
ratio = 0.7

random.seed(19941030)

file1 = open(path + '.' + str(ratio), 'w')
file2 = open(path + '.' + str(1-ratio), 'w')

for line in open(path):
    if random.random() < ratio:
        file1.write(line)
    else:
        file2.write(line)

file1.close()
file2.close()
