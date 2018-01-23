import pickle as pkl

path_model = 'data/fwfm_l2_v_1e-6_yahoo_dataset2.2_epoch_2'
path = 'data/model.txt'

d = pkl.load(open(path_model, 'rb'))
n_fields = 15
embedding_dim = 10
lst_n_feature = [len(d['w0_' + str(i)]) for i in range(15)]
total_feature = sum(lst_n_feature)

file = open(path, 'w')
file.write('==== Header ====\n')
file.write('model_version_id: 100\n')
file.write('n_field: %d\n' % n_fields)
file.write('has_bias: 0\n')
file.write('embedding_dim: %d\n' % embedding_dim)
file.write('total_feature: %d\n' % total_feature)
file.write('list_first_feature: %s\n' % ', '.join(['zero_field_' + str(x) for x in range(15)]))
file.write('list_n_feature: %s\n' % ','.join(map(str, lst_n_feature)))
file.write('==== Model ====\n')
file.write('b1: %d\n' % d['b1'])
idx_feature = 0
idx_fea = 0
for idx_field in range(15):
    lst_embedding = d['w0_%d' % idx_field]
    for i in range(len(lst_embedding)):
        file.write('feature%d: %s\n' % (idx_fea, ','.join(map(str, lst_embedding[i]))))
        idx_fea += 1
    idx_feature += 1
for idx_field in range(15):
    file.write('w_l_%d: %s\n' % (idx_field, ','.join(map(str, [x[0] for x in d['w_l'][idx_field * embedding_dim : (idx_field + 1) * embedding_dim]]))))
i = 0
for idx_field in range(15):
    for idx_field2 in range(idx_field + 1, 15):
        file.write('r_%d_%d: %s\n' % (idx_field, idx_field2, ','.join(map(str, d['w_p'][i]))))
        i += 1

cali = "====calibration====\n" + "model_version_id,line_id,start,end,calibrated_score\n" + "143,132,0.2,0.3,0.0023\n" + "143,132,0.4,0.5,0.9000\n" + "143,132,0.1,0.2,0.0012\n" + "143,132,0.5,0.6,0.0045\n" + "143,132,0.3,0.4,0.0034\n" +"-1,132,0.1,0.2,0.0056\n" +"-1,132,0.2,0.3,0.0067\n" +"-1,-1,0.1,0.2,0.0052\n" +"-1,-1,0.2,0.3,0.0069\n" +"143,-1,0.1,0.2,0.0026\n" +"143,-1,0.2,0.3,0.0017\n" +"-1,3,0.3,0.4,0.0078\n"
file.write(cali);
file.close()
