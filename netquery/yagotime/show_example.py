import pickle

# geo_file = '../../model_dir/yagotime/2023-11-11-12-33-11Time2VecEncode--fm_prec_geo_Test______add.pkl'
# file = '../../model_dir/yagotime/2023-11-11-12-33-11Time2VecEncode--fm_prec_Test______add.pkl'
# geo_file = '../../model_dir/yagotime/2023-11-12-09-57-45no--fm_prec_geo_Test______add.pkl'
# file = '../../model_dir/yagotime/2023-11-12-09-57-45no--fm_prec_geo_Test______add.pkl'
geo_file = 'G:\\xubing\\14time\se-kge-master\graphqa\data\\train_queries_2.pkl'
# file = 'G:\\xubing\\14time\se-kge-master\graphqa\data\\train_edges.pkl'


def read(path):
    return pickle.load(open(path,'rb'))

# r = read(file)
geo_r = read(geo_file)

# for key,val in c.items():
#
import json

# a = read('G:\\xubing\\14time\se-kge-master\graphqa\data\graph_data.pkl')

entity = json.load(open('G:\\xubing\\14time\se-kge-master\graphqa\data\id2entity.json','r',encoding='utf-8'))
rel = json.load(open('G:\\xubing\\14time\se-kge-master\graphqa\data\\id2rel.json','r',encoding='utf-8'))
res = []
for i in geo_r:
    tmp = []
    for j in i[0][1:]:
        tmp.append([entity[str(j[0])].strip('http://yago-knowledge.org/resource/'),rel[str(j[1][1])].strip('http://yago-knowledge.org/resource/'),entity[str(j[2])].strip('http://yago-knowledge.org/resource/')])
    res.append(tmp)
print()


