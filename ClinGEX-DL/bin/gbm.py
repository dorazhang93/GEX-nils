from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
import pickle as pkl
from util import *
from env import *
from data import Dataset
import zero
from metrics import calculate_metrics, make_summary
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


if D.is_regression:
    model = GradientBoostingRegressor(**args['model'])
    predict = model.predict
else:
    model = GradientBoostingClassifier(**args['model'])
    predict = model.predict_proba



timer = zero.Timer()
timer.run()

model.fit(
    X[TRAIN],
    Y[TRAIN],
)


dump_pickle(model, output / 'model.pickle')

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
    stats['metrics'][part] = calculate_metrics(
        D.info['task_type'], Y[part], p, 'probs'
    )
    save_prediction(part,p,output)


stats['time'] = format_seconds(timer())
dump_stats(stats, output, True)
backup_output(output)


