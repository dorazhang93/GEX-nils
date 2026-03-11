import argparse
from transformer import Transformer
from deep import *
from pathlib import Path
from scipy import special
from util import load_toml, get_categories
from env import *
from data import Dataset, to_tensors
import numpy as np
from metrics import calculate_metrics, make_summary
from functools import partial
from deep import IndexLoader
import pickle
# import os
# os.environ['CUDA_VISIBLE_DEVICES']='0'


def parse_args():
    parser=argparse.ArgumentParser(description='test a transformer model')
    parser.add_argument('--dataset', default='/home/avesta/daqu/Projects/GEX/GEX_processed/modeling_data/gex/',help="path to dataset")
    parser.add_argument('--task', default='SLNM@status', help="5-year@DRF@status LNM@status  Multifocality  SLNM@status  Tumor@size")
    parser.add_argument('--output', default='/home/avesta/daqu/Projects/GEX/code/ClinTab-DL/output/gex_midtrain/',help="path to ckpt")
    parser.add_argument('--config', default='/home/avesta/daqu/Projects/GEX/code/ClinTab-DL/configs/gex_transformer.toml')
    parser.add_argument('--repeat',default=0, type=int)
    parser.add_argument('--kfold', default=0, type=int, help='0-4')
    args=parser.parse_args()
    return args


def setup_data(D, task):
    D.task_type()
    y_idx = D.info['task_idx']
    Y = {'test': D.test['Y']['data'][:, y_idx],
         'val': D.val['Y']['data'][:, y_idx],
         'train': D.train['Y']['data'][:, y_idx],}
    Y = to_tensors(Y)
    n_classes = 1
    y_info = {'endpoint': task, 'task_type': D.info['task_type'], 'task_idx': y_idx, 'n_classes': n_classes}
    print(y_info)
    print(D.info['C'])
    # construct X=(N,C)
    X_num = {'test': D.test['N']['data'], 'val': D.val['N']['data'],'train': D.train['N']['data'],} if D.test['N']['data'] is not None else None
    X_cat = {'test': D.test['C']['data'], 'val': D.val['C']['data'],'train':D.train['C']['data'],} if D.test['C']['data'] is not None else None
    X_raw = (X_num, X_cat)
    X = tuple(None if x is None else to_tensors(x) for x in X_raw)

    case_ids = {'test': D.test['case_ids']['data'], 'val': D.val['case_ids']['data'],}
    # locate data to cuda device if gpu is available
    device = get_device()
    if device.type != 'cpu':
        X = tuple(
            None if x is None else {k: v.to(device) for k, v in x.items()} for x in X
        )
        Y = {k: v.to(device) for k, v in Y.items()}

    X_num, X_cat = X
    del X
    X_cat = {k: v.int() for k, v in X_cat.items()} if X_cat is not None else None
    if not D.is_multiclass:
        Y = {k: v.float() for k, v in Y.items()}

    return X_num, X_cat, Y, case_ids

def load_model(config, d_num, num_categories):
    device = get_device()
    config['model'].setdefault('token_bias', True)
    config['model'].setdefault('kv_compression', None)  # TODO delete if not used
    config['model'].setdefault('kv_compression_sharing', None)  # TODO delete if not used
    model = Transformer(
        d_numerical=d_num,
        categories=num_categories,
        d_out=1,
        **config['model'],
    ).to(device)
    return model

def save_prediction(split,preds, output,subtype):
    labels = Y[split]
    fortnrs = D.val['case_ids']['data'] if split == 'val' else D.test['case_ids']['data']
    assert len(preds) == len(labels)
    assert len(preds) == len(fortnrs)
    data = []
    for i in range(len(preds)):
        data.append({'case_id': fortnrs[i],
                     'pred_score': torch.tensor([preds[i]]),
                     'gt_label': torch.tensor([labels[i]])})
    with open(output / f'{split}_{subtype}.pkl', 'wb') as file:
        pickle.dump(data, file)


if __name__=='__main__':
    args = parse_args()
    output = Path(args.output)
    config = load_toml(args.config)
    subtype=args.dataset.split("_")[-1]

    # %% set up dataset
    dataset_dir = get_data_path(args.dataset) / str(args.repeat) / str(args.kfold)
    D = Dataset.from_dir(dataset_dir,args.task.replace('@', ' '))
    X_num, X_cat, Y, case_ids= setup_data(D,args.task)

    # %% load model
    model = load_model(config, X_num['test'].shape[1], get_categories(X_cat))
    checkpoint_path = output / 'checkpoint.pt'
    model.load_state_dict(torch.load(checkpoint_path)['model'])
    model.eval()


    @torch.no_grad()
    def apply_model(part, idx):
        return model(
            None if X_num is None else X_num[part][idx],
            None if X_cat is None else X_cat[part][idx],
        )

    @torch.no_grad()
    def evaluate(parts):
        eval_batch_size = 32
        device = get_device()
        model.eval()
        metrics = {}
        predictions = {}
        for part in parts:
            while eval_batch_size:
                try:
                    predictions[part] = (
                        torch.cat(
                            [
                                apply_model(part, idx)
                                for idx in IndexLoader(
                                D.size(part), eval_batch_size, False, device
                            )
                            ]
                        )
                    )
                except RuntimeError as err:
                    if not is_oom_exception(err):
                        raise
                    eval_batch_size //= 2
                    print('New eval batch size:', eval_batch_size)
                else:
                    break
            if not eval_batch_size:
                RuntimeError('Not enough memory even for eval_batch_size=1')

            # calculate pformance metrics
            nan_mask = np.isnan(Y[part].cpu().numpy())
            Y_masked = Y[part][~nan_mask]
            predictions_masked = predictions[part][~nan_mask]

            predictions[part] = predictions[part].cpu()
            metrics[part] = calculate_metrics(
                'binaryClass',
                Y_masked.cpu().numpy(),  # type: ignore[code]
                predictions_masked.cpu().numpy(),  # type: ignore[code]
                'logits',
            )

        for part, part_metrics in metrics.items():
            print(f'[{part:<5}]', make_summary(part_metrics))
        return metrics, predictions

    _, predictions = evaluate(['val', 'test'])
    for k, v in predictions.items():
        save_prediction(k, v, output,subtype)
    print('Done!')
