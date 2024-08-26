import pickle

path_auc_geo_Test = '../../model_dir/yagotime/2023-11-10-19-48-52Time2VecEncode--fm_auc_geo_Test.pkl'
path_auc_geo_Valid = '../../model_dir/yagotime/2023-11-10-19-48-52Time2VecEncode--fm_prec_geo_Test.pkl'
# path_auc_Test = '../../model_dir/spa_sem_lift_test/2023-10-28-13-27-00time_forward_NN--fm_auc_Test.pkl'
# path_auc_Valid = '../../model_dir/spa_sem_lift_test/2023-10-28-13-27-00time_forward_NN--fm_auc_Valid.pkl'

a = '../../model_dir/yagotime/2023-11-10-21-48-45Time2VecEncode--fm_prec_geo_Test______add.pkl'

def read(path):
    return pickle.load(open(path,'rb'))


auc_geo_Test = read(path_auc_geo_Test)
auc_geo_Valid = read(path_auc_geo_Valid)

c = read(a)
# auc_Test = read(path_auc_Test)
# auc_Valid = read(path_auc_Valid)
print()
