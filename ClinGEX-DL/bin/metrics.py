import typing as ty

import numpy as np
import scipy.special
import sklearn.metrics as skm
from scipy.stats import pearsonr
import util


def calculate_metrics(
    task_type: str,
    y: np.ndarray,
    prediction: np.ndarray,
    classification_mode: str,
) -> ty.Dict[str, float]:
    pos_num=sum(y)
    neg_num=len(y)
    pos_fraction= pos_num/neg_num
    if task_type == util.REGRESSION:
        del classification_mode
        rmse = skm.mean_squared_error(y, prediction,squared=False)  # type: ignore[code]
        r2_score = skm.r2_score(y,prediction)
        prr = pearsonr(y,prediction)[0]
        return {'rmse': float(rmse), 'score': float(-rmse),'r2_score':float(r2_score),'pearsonr':float(prr)}
    else:
        assert task_type in (util.BINCLASS, util.MULTICLASS)
        labels = None
        if classification_mode == 'probs':
            probs = prediction
        elif classification_mode == 'logits':
            probs = (
                scipy.special.expit(prediction)
                if task_type == util.BINCLASS
                else scipy.special.softmax(prediction, axis=1)
            )
        else:
            assert classification_mode == 'labels'
            probs = None
            labels = prediction
        if labels is None:
            labels = (
                np.round(probs).astype('int64')
                if task_type == util.BINCLASS
                else probs.argmax(axis=1)  # type: ignore[code]
            )

        result = skm.classification_report(y, labels, output_dict=True)  # type: ignore[code]
        if task_type == util.BINCLASS:
            result['roc_auc'] = skm.roc_auc_score(y, probs)  # type: ignore[code]
            result['pr_auc'] = skm.average_precision_score(y,probs)
        result['score'] = (result['roc_auc']/0.5 + result['pr_auc']/pos_fraction)/2  # type: ignore[code]
        result['pos_fraction'] = pos_fraction
    return result  # type: ignore[code]


def make_summary(metrics: ty.Dict[str, ty.Any]) -> str:
    precision = 3
    summary = {}
    for k, v in metrics.items():
        if k.isdigit():
            continue
        k = {
            'score': 'SCORE',
            'accuracy': 'acc',
            'roc_auc': 'roc_auc',
            'macro avg': 'm',
            'weighted avg': 'w',
        }.get(k, k)
        if isinstance(v, float) or isinstance(v, np.float32):
            v = round(float(v), precision)
            summary[k] = v
        else:
            v = {
                {'precision': 'p', 'recall': 'r', 'f1-score': 'f1', 'support': 's'}.get(
                    x, x
                ): round(v[x], precision)
                for x in v
            }
            for item in v.items():
                summary[k + item[0]] = item[1]

    s = [f'score = {summary.pop("SCORE"):.3f}']
    for k, v in summary.items():
        if k not in ['mp', 'mr', 'wp', 'wr']:  # just to save screen space
            s.append(f'{k} = {v}')
    return ' | '.join(s)
