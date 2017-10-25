import random

# Split the criteo training data to train/validation/test data sets.
path = '../data_cretio/train.txt'
train_ratio = 0.6
validation_ratio = 0.2

random.seed(19941030)

file1 = open(path + '.train', 'w')
file2 = open(path + '.validation', 'w')
file3 = open(path + '.test', 'w')

batch = 100000
cnt_total = 45840617

for i, line in enumerate(open(path)):
    if i % batch == batch - 1:
        print i * 1.0 / cnt_total
    ran = random.random()
    if ran < train_ratio:
        file1.write(line)
    elif ran < train_ratio + validation_ratio:
        file2.write(line)
    else:
        file3.write(line)

file1.close()
file2.close()
