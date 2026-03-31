import pickle
import numpy as np
rng=np.random.default_rng()

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent

OUT_DIR = PROJECT_DIR / "Data"

OUT_DIR.mkdir(parents=True, exist_ok=True)

"""  The format of packed input data
data_packed={'test':{'N':{'data':test_N_data,},
                       'C':{'data':test_C_data,},
                       'Y':{'data':test_Y_data},
                         'case_ids':{'data':test_ids}},
            'val':{'N':{'data':val_N_data,},
                       'C':{'data':val_C_data,},
                       'Y':{'data':val_Y_data},
                      'case_ids':{'data':val_ids}},
            'train':{'N':{'data':train_N_data,},
                       'C':{'data':train_C_data,},
                       'Y':{'data':train_Y_data},
                      'case_ids':{'data':train_ids}},
            'info':{'N':N_info,'C':C_info,'Y':Y_info}
    }
"""
n_sample_test=1000
n_sample_val=1000
n_sample_train=2500

"""
load meta information of numerical (N), categorical (C) variables, and ground-truth (Y) labels
info:{
    'N':{'index':'variable name'},
    'C':{'index':'variable name'),
    'Y':{'regression':{'task name': 'task index'},
         'binary_cls':{'task name': 'task index'},
         'multi_cls':{'task name': 'task index'},
         }
}
"""
with open(OUT_DIR/'info.pickle','rb') as f:
    info_meta=pickle.load(f)

def synthesis_N(n_sample, n_feature):
    mu=0
    var=1
    return rng.normal(mu, var, size=(n_sample,n_feature))

def synthesis_C(n_sample, n_feature):
    categorical_distributions={'0':{'a':[0,1,2], 'p':[0.2,0.6,0.2]},
                               '1':{'a':[0,1], 'p':[0.4,0.6]},
                               '2': {'a': [0, 1, 2], 'p': [0.2, 0.6, 0.2]},
                               '3': {'a': [0, 1], 'p': [0.4, 0.6,]},
                               '4': {'a': [0, 1], 'p': [0.12, 0.88]},
                               '5': {'a': [0, 1, 2, 3], 'p': [0.55, 0.2, 0.11, 0.14]},
                               }
    C_data=[]
    assert n_feature==6
    for i in range(n_feature):
        C_data.append(rng.choice(**(categorical_distributions[str(i)]), size=(n_sample,1)))


    return np.concatenate(C_data, axis=1).astype(np.float)

def synthesis_Y(n_sample, task_info):
    task_list_sorted=['Tumor size','No. LNMs','Multifocality','LNM status',
                      'SLNM status','5-year survival status','5-year RF status',
                      '5-year DRF status','No. invasive foci','SLNM type']
    categorical_dists={
                       'Multifocality':{'a':[0,1], 'p':[0.88,0.12,]},
                       'LNM status':{'a':[0,1], 'p':[0.78,0.22,]},
                      'SLNM status':{'a':[0,1], 'p':[0.84,0.16,]},
                       '5-year survival status':{'a':[0,1], 'p':[0.84,0.16,]},
                       '5-year RF status':{'a':[0,1], 'p':[0.84,0.16,]},
                      '5-year DRF status':{'a':[0,1], 'p':[0.94,0.06,]},
                       'No. invasive foci':{'a':[0,1,2], 'p':[0.80,0.16,0.04]},
                       'SLNM type':{'a':[0,1,2], 'p':[0.80,0.16,0.04]},}
    mu=0
    var=1
    tsize_NoLNMs_data = rng.normal(mu, var, size=(n_sample,2))
    C_Y_data=[tsize_NoLNMs_data]
    for task in task_list_sorted[2:]:
        C_Y_data.append(rng.choice(**(categorical_dists[task]), size=(n_sample,1)))


    return np.concatenate(C_Y_data, axis=1).astype(np.float)


"""
code for generating an example dataset with a training set (n=2500), a validation set (n=1000) and a test set (n=1000)

"""
train_data = {'N': synthesis_N(n_sample_train,n_feature=len(info_meta['N'])),
              'C': synthesis_C(n_sample_train, n_feature=len(info_meta['C'])),
              'Y': synthesis_Y(n_sample_train, task_info=info_meta['Y'])}


val_data = {'N': synthesis_N(n_sample_val,n_feature=len(info_meta['N'])),
              'C': synthesis_C(n_sample_val, n_feature=len(info_meta['C'])),
              'Y': synthesis_Y(n_sample_val, task_info=info_meta['Y'])}

test_data = {'N': synthesis_N(n_sample_test,n_feature=len(info_meta['N'])),
              'C': synthesis_C(n_sample_test, n_feature=len(info_meta['C'])),
              'Y': synthesis_Y(n_sample_test, task_info=info_meta['Y'])}

data_packed={'test':{'N':{'data':test_data['N'],},
                       'C':{'data':test_data['C'],},
                       'Y':{'data':test_data['Y']},
                         'case_ids':{'data':np.array([ 'Test'+str(x) for x in range(n_sample_test)])}},
            'val':{'N':{'data':val_data['N'],},
                       'C':{'data':val_data['C'],},
                       'Y':{'data':val_data['Y'],},
                      'case_ids':{'data':np.array([ 'Val'+str(x) for x in range(n_sample_val)])}},
            'train':{'N':{'data':train_data['N'],},
                       'C':{'data':train_data['C'],},
                       'Y':{'data':train_data['Y'],},
                      'case_ids':{'data':np.array([ 'Train'+str(x) for x in range(n_sample_train)])}},
            'info':info_meta,
    }

output_folder = OUT_DIR/'synthesis_cancerpathway_GEX'/'0'/'0'
output_folder.mkdir(parents=True, exist_ok=True)
with open(output_folder / "build_X_Y.pickle","wb") as f:
    pickle.dump(data_packed,f)

