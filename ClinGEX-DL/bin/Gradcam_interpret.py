import argparse
import pytorch_grad_cam as cam
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients
from transformer import Transformer
# from transformer_transfer import Transformer
from deep import *
from pathlib import Path
from scipy import special
from util import load_toml, get_categories
from env import *
from data import Dataset, to_tensors
import numpy as np
from metrics import calculate_metrics
from functools import partial
from deep import IndexLoader
import pickle
# import os
# os.environ['CUDA_VISIBLE_DEVICES']='0'


def parse_args():
    parser=argparse.ArgumentParser(description='test a transformer model')
    parser.add_argument('--dataset', default='/home/avesta/daqu/Projects/GEX/GEX_processed/modeling_data/gex/',help="path to dataset")
    parser.add_argument('--split', default='val',help="val or test ")
    parser.add_argument('--task', default='SLNM@status', help="5-year@DRF@status LNM@status  Multifocality  SLNM@status  Tumor@size")
    parser.add_argument('--work-dir', default='/home/avesta/daqu/Projects/GEX/code/ClinTab-DL/output/gex_midtrain/',help="path to ckpt")
    parser.add_argument('--model', default='transformer', help="transformer")
    parser.add_argument('--target-layers', default=['layers.0.attention.attend','tokenizer']) #'to_embedding' or 'tokenizer'
    parser.add_argument('--config', default='/home/avesta/daqu/Projects/GEX/code/ClinTab-DL/configs/gex_transformer.toml')
    parser.add_argument('--repeat',default=0, type=int)
    parser.add_argument('--kfold', default=0, type=int, help='0-4')
    args=parser.parse_args()
    return args

def init_cam(model, target_layers, reshape_transform):
    cam_instance = GradCAM(model, target_layers, use_cuda=True)
    # Release the original hooks in ActivationsAndGradients to use
    # ActivationsAndGradients.
    cam_instance.activations_and_grads.release()
    cam_instance.activations_and_grads = ActivationsAndGradients(
        cam_instance.model, cam_instance.target_layers, reshape_transform=reshape_transform)

    return cam_instance

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
    # model = Transformer(
    #     categories=num_categories,
    #     d_out=1,
    #     **config['model'],
    # ).to(device)
    model = Transformer(
        d_numerical=d_num,
        categories=num_categories,
        d_out=1,
        **config['model'],
    ).to(device)
    return model

def sanity_check(model, x_Num,X_cat, Y):
    device=get_device()
    @torch.no_grad()
    def apply_model(idx):
        return model(
            None if x_Num is None else x_Num['val'][idx],
            None if X_cat is None else X_cat['val'][idx],
        )

    predictions = torch.cat(
        [
            apply_model(idx)
            for idx in IndexLoader(
            len(X_num['val']),
            32, False, device
        )
        ]
    )

    # calculate pformance metrics
    nan_mask = np.isnan(Y['val'].cpu().numpy())
    Y_masked = Y['val'][~nan_mask]
    predictions_masked = predictions[~nan_mask]
    result = calculate_metrics(
        D.info['task_type'],
        Y_masked.cpu().numpy(),  # type: ignore[code]
        predictions_masked.cpu().numpy(),  # type: ignore[code]
        'logits',
    )
    print(result)


def get_target_layer(target_layer, model):
    # preview the model
    for name, layer in model.named_modules():
        # print(name, layer)
        if name == target_layer:
            return layer
    raise AttributeError(
        f'Cannot get the layer "{target_layer}". Please choose from: \n' +
        '\n'.join(name for name, _ in model.named_modules()))

def reshape_transform(tensor, model, args):
    #'layers.0.attention.attend' : activation shape = batchsize,n_head, 1, num_genes
    # 'tokenizer' : activation shape = batchsize, num_genes, d_token

    if tensor.shape[2] == 1:
        tensor = tensor[:,:,0,:].transpose(1,2)
    return tensor

if __name__=='__main__':
    args = parse_args()
    output = Path(args.work_dir+'/'+args.task +'/'+args.model+'/'+str(args.repeat) +'/'+ str(args.kfold))
    config = load_toml(args.config)

    # %% set up dataset
    dataset_dir = get_data_path(args.dataset) / str(args.repeat) / str(args.kfold)
    D = Dataset.from_dir(dataset_dir,args.task.replace('@', ' '))
    X_num, X_cat, Y, case_ids= setup_data(D,args.task)
    data_size=len(X_num[args.split])

    # %% load model
    model = load_model(config, X_num['test'].shape[1], get_categories(X_cat))
    checkpoint_path = output / 'checkpoint.pt'
    model.load_state_dict(torch.load(checkpoint_path)['model'])
    model.eval()

    sanity_check(model, X_num, X_cat, Y)

    # %% get target layers
    target_layers = [
        get_target_layer(layer_str, model) for layer_str in args.target_layers
    ]

    # %% initialize cam
    cam_ = init_cam(model, target_layers, partial(reshape_transform, model=model, args=args))

    # calculate cam grads
    case_predictions=[]
    feature_importantce=[]
    for idx in IndexLoader(train_size=data_size, batch_size=32, shuffle=False, device=get_device()):
        grayscale_cam, predictions = cam_(
            (X_num[args.split][idx],X_cat[args.split][idx] if X_cat is not None else None),
            target_category=0,)
        batch_caseids=case_ids[args.split][idx.cpu().numpy()]
        batch_gtTruth=Y[args.split][idx].cpu().numpy()
        assert len(batch_caseids)==len(predictions)

        for case_id, gt, predict, cam_value in zip(batch_caseids,batch_gtTruth, predictions,grayscale_cam):
            dev_tsize_mean, dev_tsize_std = 17.8, 8.2  # Gex
            if args.task == 'Tumor@size':
                predict = special.expit(predict)
            else:
                predict = predict * dev_tsize_std + dev_tsize_mean
            case_predictions.append([case_id,gt,predict]) # case id , ground truth, prediction
            feature_importantce.append(cam_value)

    with open(output / f"{args.split}_cam_feature_importance.pickle", 'wb') as fout:
        pickle.dump({'case_prediction':np.array(case_predictions),
                     'cam_value':np.array(feature_importantce)}, fout)




