import numpy as np
from xgboost import XGBClassifier, XGBRegressor
import pickle as pkl
from util import *
from env import *
from data import Dataset
import zero
from metrics import calculate_metrics
import pandas as pd

TRAIN = 'train'
VAL = 'val'
TEST = 'test'
PARTS = [TRAIN, VAL, TEST]

args, output = load_config()


# %%
zero.set_randomness(args['repeat'])
dataset_dir = get_data_path(args['dataset']) / str(args["repeat"]) / str(args["kfold"])
stats: ty.Dict[str, ty.Any] = {
    'dataset': dataset_dir.name,
    'algorithm': Path(__file__).stem,
    **load_json(output / 'stats.json'),
}

# Prepare data and model
D = Dataset.from_dir(dataset_dir,args["endpoint"])

# extract Y data of the endpoint
D.task_type()
D.ohe_cat()
D.filter_train_nan()
y_idx=D.info['task_idx']
Y={'test':D.test['Y']['data'][:,y_idx],
   'val':D.val['Y']['data'][:,y_idx],
   'train':D.train['Y']['data'][:,y_idx]}
n_classes = len(set(Y[TRAIN].tolist()))if D.is_multiclass else 1
y_info={'endpoint':args['endpoint'], 'task_type':D.info['task_type'], 'task_idx': y_idx, 'n_classes':n_classes}
print(y_info)
# construct X
X_num = {'test':D.test['N']['data'],'val':D.val['N']['data'],'train':D.train['N']['data']} if D.test['N']['data'] is not None else None
X_cat = {'test':D.test['C']['data'],'val':D.val['C']['data'],'train':D.train['C']['data']} if D.test['C']['data'] is not None else None
X_cat = {k:v.astype(int) for k,v in X_cat.items()} if X_cat is not None else None

# filter label=nan
nan_mask_Y={k: ~np.isnan(v) for k,v in Y.items()}
Y_filter_nan = {k:v[nan_mask_Y[k]] for k,v in Y.items()}
X_num_filter_nan ={k:v[nan_mask_Y[k]] for k,v in X_num.items()} if X_num is not None else None
X_cat_filter_nan ={k:v[nan_mask_Y[k]] for k,v in X_cat.items()} if X_cat is not None else None

n_num_features = 0 if X_num is None else X_num[TRAIN].shape[1]
n_cat_features = 0 if X_cat is None else X_cat[TRAIN].shape[1]
n_features = n_num_features + n_cat_features
if X_num is None:
    assert X_cat is not None
    X = {x: pd.DataFrame(X_cat[x], columns=range(n_features)) for x in X_cat}
elif X_cat is None:
    assert X_num is not None
    X = {x: pd.DataFrame(X_num[x], columns=range(n_features)) for x in X_num}
else:
    X = {
        k: pd.concat(
            [
                pd.DataFrame(X_num[k], columns=range(n_num_features)),
                pd.DataFrame(X_cat[k], columns=range(n_num_features, n_features)),
            ],
            axis=1,
        )
        for k in X_num.keys()
    }

if X_num_filter_nan is None:
    assert X_cat_filter_nan is not None
    X_filter_nan = {x: pd.DataFrame(X_cat_filter_nan[x], columns=range(n_features)) for x in X_cat_filter_nan}
elif X_cat_filter_nan is None:
    assert X_num_filter_nan is not None
    X_filter_nan = {x: pd.DataFrame(X_num_filter_nan[x], columns=range(n_features)) for x in X_num_filter_nan}
else:
    X_filter_nan = {
        k: pd.concat(
            [
                pd.DataFrame(X_num_filter_nan[k], columns=range(n_num_features)),
                pd.DataFrame(X_cat_filter_nan[k], columns=range(n_num_features, n_features)),
            ],
            axis=1,
        )
        for k in X_num_filter_nan.keys()
    }

model_kwargs = args['model']

# XGBoost does not automatically use the best model, so early stopping must be used
assert 'early_stopping_rounds' in model_kwargs
if D.is_regression:
    model = XGBRegressor(**model_kwargs)
    predict = model.predict

else:
    model = XGBClassifier(**model_kwargs, disable_default_eval_metric=True, eval_metric='auc')
    predict = lambda x: model.predict_proba(x)[:, 1]  # type: ignore[code]  # noqa


timer = zero.Timer()
timer.run()

model.fit(
    X[TRAIN],
    Y[TRAIN],
    eval_set=[(X_filter_nan[VAL], Y_filter_nan[VAL])],
    verbose=True
)


model.save_model(str(output / 'model.json'))

def save_prediction(split,preds, output):
    labels = Y[split]
    fortnrs = D.val['case_ids']['data'] if split == 'val' else D.train['case_ids']['data']\
        if split=='train' else D.test['case_ids']['data']
    assert len(preds) == len(labels)
    assert len(preds) == len(fortnrs)
    data = []
    for i in range(len(preds)):
        data.append({'case_id': fortnrs[i],
                     'pred_score': torch.tensor([preds[i]]),
                     'gt_label': torch.tensor([labels[i]])})
    with open(output / f'{split}.pkl', 'wb') as file:
        pkl.dump(data, file)

stats['metrics'] = {}
for part in X:
    p = predict(X[part])
    nan_mask = ~np.isnan(Y[part])
    stats['metrics'][part] = calculate_metrics(
        D.info['task_type'], Y[part][nan_mask], p[nan_mask], 'probs'
    )
    save_prediction(part,p,output)


stats['time'] = format_seconds(timer())
dump_stats(stats, output, True)
backup_output(output)


