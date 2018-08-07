d_index_conv_type = {
    '34884': 'View_Content',
    '34885': 'Purchase',
    '34886': 'Sign_Up',
    '34887': 'Lead'
}
path_data = '../data_cvr2/cvr_imp_20180722_0728_conv_20180722_0803.csv.add_conv_type.thres5.yx'

def split_by_conv_type(path):
    d_index_file = {}
    for index in d_index_conv_type.keys():
        d_index_file[index] = open(path+'.'+d_index_conv_type[index], 'w')
    for line in open(path):
        index = line.strip('\n').split(' ')[-1].split(':')[0]
        d_index_file[index].write(line)
    for index in d_index_file.keys():
        d_index_file[index].close()

split_by_conv_type(path_data)