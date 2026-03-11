import numpy as np
import scipy.special
import torch
from deep import *
import argparse
import shap
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from util import load_toml
from transformer import Transformer
from env import *
from data import Dataset, to_tensors
from metrics import calculate_metrics

import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

class MLP(nn.Module):
    def __init__(
        self,
        *,
        d_in: int,
        d_layers: ty.List[int],
        dropout: float,
        d_out: int,
        categories: ty.Optional[ty.List[int]],
        d_embedding: int,
    ) -> None:
        super().__init__()

        if categories is not None:
            # d_in += len(categories) * d_embedding # for cat vars tokenizer TODO
            d_in+=len(categories) # for ordinal encoding and one-hot-encoding
            category_offsets = torch.tensor([0] + categories[:-1]).cumsum(0)
            self.register_buffer('category_offsets', category_offsets)
            self.category_embeddings = nn.Embedding(sum(categories), d_embedding)
            nn.init.kaiming_uniform_(self.category_embeddings.weight, a=math.sqrt(5))
            print(f'{self.category_embeddings.weight.shape=}')

        self.layers = nn.ModuleList(
            [
                nn.Linear(d_layers[i - 1] if i else d_in, x)
                for i, x in enumerate(d_layers)
            ]
        )
        self.dropout = dropout
        self.head = nn.Linear(d_layers[-1] if d_layers else d_in, d_out)

    def forward(self, x_num, x_cat):
        x = []
        if x_num is not None:
            x.append(x_num)
        if x_cat is not None:
            x.append(x_cat)

        x = torch.cat(x, dim=-1).float()

        for layer in self.layers:
            x = layer(x)
            x = F.relu(x)
            if self.dropout:
                x = F.dropout(x, self.dropout, self.training)
        x = self.head(x)
        x = x.squeeze(-1)
        return x

def plot_shap_importance(df_importance,output):
    plt.subplots(1,1,gridspec_kw={'left':0.28,'right':0.97})
    sns.set(font_scale=1.3)
    plt.grid(linestyle='dotted')
    plt.figure(figsize=(10,12))
    sns.barplot(data=df_importance, orient="h")
    plt.axvline(x=0, color=".5")
    plt.title("shap feature importance")
    plt.xlabel("shap feature importance")
    plt.subplots_adjust(left=0.3)
    plt.savefig(output)

def plot_shap_feature_importance(shap_values, output,columns):
    feature_importance=np.abs(shap_values).mean(axis=0)
    # feature_importance=feature_importance/feature_importance.max()
    feature_importance=pd.DataFrame([feature_importance],columns=columns)
    columns_new=np.array(columns)[np.argsort(feature_importance.values)[0][::-1]]
    feature_importance=feature_importance[columns_new]
    plot_shap_importance(feature_importance,output)
    return

def calc_shap(model,X,x_vars,output,split):
    columns= [x_vars[str(i)] for i in range(len(x_vars))]
    # x_init = np.concatenate([X[0]["init"],X[1]["init"]],axis=1)
    # x_tgt = np.concatenate([X[0]["target"],X[1]["target"]],axis=1)
    x_init=X[0]["init"]
    x_tgt =X[0]['target']
    x_init=pd.DataFrame(x_init, columns=columns)
    x_tgt=pd.DataFrame(x_tgt, columns=columns)
    # explainer = shap.Explainer(model,x_init)
    explainer = shap.KernelExplainer(model, shap.kmeans(x_init,50))
    shap_values = explainer(x_tgt)
    np.save(output / f'{split}_shap_values.npy', shap_values.values)
    plot_shap_feature_importance(shap_values.values, output / "feature_importance.png", columns)

    return
def load_subtypes():
    ssp_data = pd.read_csv(
        "/home/avesta/daqu/Projects/GEX/LN_cohort_v3.0_data/LN_cohort_v3.0_StringTie/LNStringTie.txt", sep='\t')
    cohort_identifier_file = '/home/avesta/daqu/Projects/GEX/LN_cohort_v3.0_data/LN_cohort_v3.0/LNcohort.txt'
    cohort_identifier = pd.read_csv(cohort_identifier_file, sep='\t')
    PatientName2CaseName = dict(zip(cohort_identifier.PatientName.values, cohort_identifier.CaseName.values))
    ssp_data['CaseName'] = [PatientName2CaseName[pid] for pid in ssp_data.PatientName.values]
    data = ssp_data[['CaseName', 'SSP_PAM50subtype']]
    data = data.rename(columns={"SSP_PAM50subtype": "subtype",
                                "CaseName": 'case_id'})
    data = data.replace({"subtype":{'Basal':'TNBC',
                                    'LumA':'ER+HER2-',
                                    'LumB':'ER+HER2-',
                                    'Normal':'ER+HER2-',
                                    'Her2':'HER2+'}})
    return data


def select_cohort(D, split, subtype):
    # shap_init data set
    Y_init = D.train['Y']['data']
    X_N_init = D.train['N']['data']
    X_C_init = D.train['C']['data']
    case_ids_init = D.train['case_ids']['data']

    # shap target data
    if split == 'test':
        Y = D.test['Y']['data']
        X_N =  D.test['N']['data']
        X_C = D.test['C']['data']
        case_ids = D.test['case_ids']['data']
    elif split == 'val':
        Y = D.val['Y']['data']
        X_N = D.val['N']['data']
        X_C = D.val['C']['data']
        case_ids = D.val['case_ids']['data']
    else:
        raise ValueError("Not implemented!!!")

    Y = {'init': Y_init, 'target': Y}
    N = {'init': X_N_init, 'target': X_N}
    C = {'init': X_C_init, 'target': X_C} if X_C_init is not None else None
    case_ids = {'init': case_ids_init, 'target': case_ids}


    if subtype !='all':
        subtype_data = load_subtypes()
        case2subtyp=dict(zip(subtype_data.case_id.values, subtype_data.subtype.values))
        case_filterIn = {k: np.array([True if case2subtyp[id]==subtype else False for id in np.squeeze(v)]) for k,v in case_ids.items()}
        Y = {k:v[case_filterIn[k]] for k,v in Y.items()}
        N = {k:v[case_filterIn[k]] for k,v in N.items()}
        C = {k:v[case_filterIn[k]] for k,v in C.items()} if C is not None else None
        case_ids = {k:v[case_filterIn[k]] for k,v in case_ids.items()}

    return Y, N, C, case_ids



def parse_args():
    parser=argparse.ArgumentParser(description='test a transformer model')
    parser.add_argument('--dataset', default='/home/avesta/daqu/Projects/GEX/GEX_processed/modeling_data/gex/allgene',help="path to dataset")
    parser.add_argument('--split', default='val',help="val or test ")
    parser.add_argument('--subtype', default="all", help="all, ER+HER2-, HER2+, TNBC")
    parser.add_argument('--task', default='SLNM@status', help="5-year@DRF@status LNM@status  Multifocality  SLNM@status  Tumor@size")
    parser.add_argument('--work-dir', default='/home/avesta/daqu/Projects/GEX/code/ClinTab-DL/output/gex/top2.5k_varying_sw_norm',help="path to ckpt")
    parser.add_argument('--model', default='transformer', help="transformer, mlp")
    parser.add_argument('--config', default='/home/avesta/daqu/Projects/GEX/code/ClinTab-DL/configs/gex_transformer.toml')
    parser.add_argument('--repeat',default=0, type=int)
    parser.add_argument('--kfold', default=0, type=int, help='0-4')
    args=parser.parse_args()
    return args


if __name__=='__main__':
    args = parse_args()
    output = Path(args.work_dir+'/'+args.task +'/'+args.model+'/'+str(args.repeat) +'/'+ str(args.kfold))
    config = load_toml(args.config)

    # %% set up dataset
    dataset_dir = get_data_path(args.dataset) / str(args.repeat) / str(args.kfold)
    D = Dataset.from_dir(dataset_dir,args.task.replace('@', ' '))
    D.task_type()

    # select dev or test by subtype
    Y, X_num, X_cat, case_ids = select_cohort(D,args.split, args.subtype)

    # construct Y
    y_idx = D.info['task_idx']
    Y={k:v[:,y_idx] for k,v in Y.items()}
    Y = to_tensors(Y)
    n_classes = 1
    y_info = {'endpoint': args.task, 'task_type': D.info['task_type'], 'task_idx': y_idx, 'n_classes': n_classes}
    print(y_info)
    print(D.info['C'])

    # construct X = (N, C)
    X_raw = (X_num, X_cat)
    X = tuple(None if x is None else to_tensors(x) for x in X_raw)

    # locate data to cuda device if gpu is available
    device = torch.device('cuda:0' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu')
    if device.type != 'cpu':
        X = tuple(None if x is None else {k: v.to(device) for k, v in x.items()} for x in X)
        Y = {k: v.to(device) for k, v in Y.items()}

    if not D.is_multiclass:
        Y = {k: v.float() for k, v in Y.items()}

    x_Num, x_Cat = X
    x_Cat = {k:v.int() for k, v in x_Cat.items()} if x_Cat is not None else None

    if args.model=='transformer':
        config['model'].setdefault('token_bias', True)
        config['model'].setdefault('kv_compression', None)  # TODO delete if not used
        config['model'].setdefault('kv_compression_sharing', None)  # TODO delete if not used
        model = Transformer(
            d_numerical=0 if X_num is None else X_num['init'].shape[1],
            categories=None if X_cat is None else [len(set(X_cat['init'][:, i].tolist())) for i in
                                                   range(X_cat['init'].shape[1])],
            d_out=n_classes,
            **config['model'],
        ).to(device)
    elif args.model=='mlp':
        model = MLP(
            d_in=0 if X_num is None else X_num['init'].shape[1],
            d_out=n_classes,
            categories=None if X_cat is None else [len(set(X_cat['init'][:, i].tolist())) for i in
                                                   range(X_cat['init'].shape[1])],
            **config['model'],
        ).to(device)

    checkpoint_path = output / 'checkpoint.pt'


    def explain_ft(model, X_raw):
        model.load_state_dict(torch.load(checkpoint_path)['model'])
        model.eval()

        @torch.no_grad()
        def apply_model(idx):
            return model(
                None if x_Num['init'] is None else x_Num['init'][idx],
                None if x_Cat is None else x_Cat['init'][idx],
            )

        predictions = torch.cat(
                            [
                                apply_model(idx)
                                for idx in IndexLoader(
                                    len(X_num['init']),
                                32, False, device
                                )
                            ]
                        )

        # calculate pformance metrics
        nan_mask = np.isnan(Y['init'].cpu().numpy())
        Y_masked = Y['init'][~nan_mask]
        predictions_masked = predictions[~nan_mask]
        result = calculate_metrics(
            D.info['task_type'],
            Y_masked.cpu().numpy(),  # type: ignore[code]
            predictions_masked.cpu().numpy(),  # type: ignore[code]
            'logits',
            y_info,
        )
        print(result)

        x_vars=D.info['N']
        num_N=len(list(D.info['N'].keys()))
        if D.info['C'] is not None:
            for i, v in D.info['C'].items():
                x_vars[str(int(i)+num_N)]=v

        @torch.no_grad()
        def f(x):
            # x_num = x.iloc[:, :11].values
            # x_cat = x.iloc[:, 11:].values
            # x_num = torch.as_tensor(x_num).to(device)
            # x_cat = torch.as_tensor(x_cat).to(device).int()
            # predict = model(x_num, x_cat).detach().cpu().numpy()

            x = torch.as_tensor(x).to(device)
            predict = model(x, None).detach().cpu().numpy()
            if args.task!="Tumor@size":
                predict = (scipy.special.expit(predict))
            return predict

        calc_shap(f,X_raw,x_vars, output,args.split)

    explain_ft(model, X_raw)
