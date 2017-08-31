import random

# Split the criteo training data to train/validation/test data sets.
path = '/tmp/jwpan/data_cretio/train.txt.thres20.yx'
train_ratio = 0.6
validation_ratio = 0.2

random.seed(19941030)

file1 = open(path + '.train', 'w')
file2 = open(path + '.validation', 'w')
file3 = open(path + '.test', 'w')

for line in open(path):
    ran = random.random()
    if ran < train_ratio:
        file1.write(line)
    elif ran < train_ratio + validation_ratio:
        file2.write(line)
    else:
        file3.write(line)

file1.close()
file2.close()
