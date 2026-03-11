# %%
import pickle as pkl
from util import *
from deep import *
from env import *
from data import Dataset, to_tensors
from metrics import calculate_metrics, make_summary


TRAIN = 'train'
VAL = 'val'
TEST = 'test'
PARTS = [TRAIN, VAL, TEST]
# %%
class MLP(nn.Module):
    def __init__(
        self,
        *,
        d_in: int,
        feature_channel: int,
        d_layers: ty.List[int],
        dropout: float,
        d_out: int,
        d_embedding: int,
    ) -> None:
        super().__init__()
        self.mutation_layers = nn.ModuleList(
            [
                nn.Linear(feature_channel, 32),
                nn.Linear(32,1)
            ]
        )

        self.layers = nn.ModuleList(
            [
                nn.Linear(d_layers[i - 1] if i else d_in, x)
                for i, x in enumerate(d_layers)
            ]
        )
        self.dropout = dropout
        self.head = nn.Linear(d_layers[-1] if d_layers else d_in, d_out)

    def forward(self, x_num, x_cat):
        x=x_num[:,:,-1].float()
        # for layer in self.mutation_layers:
        #     x=F.relu(layer(x))

        x=x.reshape(len(x),-1)

        for layer in self.layers:
            x = layer(x)
            x = F.relu(x)
            if self.dropout:
                x = F.dropout(x, self.dropout, self.training)
        x = self.head(x)
        x = x.squeeze(-1)
        return x

# os.environ['CUDA_VISIBLE_DEVICES']='0'
# %%
args, output = load_config()

# %%
zero.set_randomness(args['repeat'])
dataset_dir = get_data_path(args['dataset']) / str(args["repeat"]) / str(args["kfold"])
stats: ty.Dict[str, ty.Any] = {
    'dataset': dataset_dir.name,
    'algorithm': Path(__file__).stem,
    **load_json(output / 'stats.json'),
}
timer = zero.Timer()
timer.run()

D = Dataset.from_dir(dataset_dir,args["endpoint"])

# extract Y data of the endpoint
D.task_type()
D.ohe_cat() # TODO disable if using tokenizer
D.filter_train_nan()
y_idx=D.info['task_idx']
Y={'test':D.test['Y']['data'][:,y_idx],
   'val':D.val['Y']['data'][:,y_idx],
   'train':D.train['Y']['data'][:,y_idx]}
Y = to_tensors(Y)
n_classes = len(set(Y[TRAIN].tolist()))if D.is_multiclass else 1
y_info={'endpoint':args['endpoint'], 'task_type':D.info['task_type'], 'task_idx': y_idx, 'n_classes':n_classes}
print(y_info)
# construct X=(N,C)
X_num = {'test':D.test['N']['data'],'val':D.val['N']['data'],'train':D.train['N']['data']} if D.test['N']['data'] is not None else None
X_cat = {'test':D.test['C']['data'],'val':D.val['C']['data'],'train':D.train['C']['data']} if D.test['C']['data'] is not None else None
X_raw = (X_num, X_cat)
X = tuple(None if x is None else to_tensors(x) for x in X_raw)

# locate data to cuda device if gpu is available
device = get_device()
if device.type != 'cpu':
    X = tuple(None if x is None else {k: v.to(device) for k, v in x.items()} for x in X)
    Y_device = {k: v.to(device) for k, v in Y.items()}
else:
    Y_device = Y
X_num, X_cat = X
del X
X_cat={k: v.int() for k, v in X_cat.items()} if X_cat is not None else None
if not D.is_multiclass:
    Y_device = {k: v.float() for k, v in Y_device.items()}

train_size = D.size(TRAIN)
batch_size = args['training']['batch_size']
epoch_size = stats['epoch_size'] = math.ceil(train_size / batch_size)
eval_batch_size = args['training']['eval_batch_size']

loss_fn = (
        F.binary_cross_entropy_with_logits
        if D.is_binclass
        else F.cross_entropy
        if D.is_multiclass
        else F.mse_loss
    )


model = MLP(
    d_in=0 if X_num is None else X_num['train'].shape[1],
    feature_channel=X_num['train'].shape[2],
    d_out=n_classes,
    **args['model'],
).to(device)
stats['n_parameters'] = get_n_parameters(model)
optimizer = make_optimizer(
    args['training']['optimizer'],
    model.parameters(),
    args['training']['lr'],
    args['training']['weight_decay'],
)
# try:
#     scheduler, _, _  = lib.make_lr_schedule(optimizer, args['lr_schedule'])
# except:
#     pass
stream = zero.Stream(IndexLoader(train_size, batch_size, True, device))
progress = zero.ProgressTracker(args['training']['patience'])
training_log = {TRAIN: [], VAL: [], TEST: []}
timer = zero.Timer()
checkpoint_path = output / 'checkpoint.pt'

def print_epoch_info():
    print(f'\n>>> Epoch {stream.epoch} | {format_seconds(timer())} | {output}')
    print(
        ' | '.join(
            f'{k} = {v}'
            for k, v in {
                'lr': get_lr(optimizer),
                'batch_size': batch_size,
                'epoch_size': stats['epoch_size'],
                'n_parameters': stats['n_parameters'],
            }.items()
        )
    )


@torch.no_grad()
def evaluate(parts):
    model.eval()
    metrics = {}
    predictions = {}
    for part in parts:
        predictions[part] = (
            torch.cat(
                [
                    model(
                        None if X_num is None else X_num[part][idx],
                        None if X_cat is None else X_cat[part][idx],
                    )
                    for idx in IndexLoader(
                        D.size(part),
                        eval_batch_size,
                        False,
                        device,
                    )
                ]
            )
        )

        nan_mask=np.isnan(Y_device[part].cpu().numpy())
        Y_masked=Y_device[part][~nan_mask]
        predictions_masked=predictions[part][~nan_mask]

        loss = loss_fn(predictions_masked,Y_masked)
        predictions[part] = predictions[part].cpu()
        metrics[part] = calculate_metrics(
            D.info['task_type'],
            Y_masked.cpu().numpy(),  # type: ignore[code]
            predictions_masked.cpu().numpy(),  # type: ignore[code]
            'logits',
        )
        metrics[part]['loss'] = loss.cpu().numpy().item()


    for part, part_metrics in metrics.items():
        print(f'[{part:<5}]', make_summary(part_metrics))
    return metrics, predictions


def save_checkpoint(final):
    torch.save(
        {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'stream': stream.state_dict(),
            'random_state': zero.get_random_state(),
            **{
                x: globals()[x]
                for x in [
                    'progress',
                    'stats',
                    'timer',
                    'training_log',
                ]
            },
        },
        checkpoint_path,
    )
    dump_stats(stats, output, final)
    backup_output(output)


# %%
timer.run()
for epoch in stream.epochs(args['training']['n_epochs']):
    print_epoch_info()

    model.train()
    epoch_losses = []
    for batch_idx in epoch:
        optimizer.zero_grad()

        loss = loss_fn(
            model(
                None if X_num is None else X_num[TRAIN][batch_idx],
                None if X_cat is None else X_cat[TRAIN][batch_idx],
            ),
            Y_device[TRAIN][batch_idx],
        )
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.detach())
    # scheduler.step()
    epoch_losses = torch.stack(epoch_losses).tolist()
    training_log[TRAIN].extend(epoch_losses)
    print(f'[{TRAIN}] loss = {round(sum(epoch_losses) / len(epoch_losses), 3)}')

    metrics, predictions = evaluate([VAL, TEST])
    for k, v in metrics.items():
        training_log[k].append(v)
    progress.update(metrics[VAL]['score'])

    if progress.success:
        print('New best epoch!')
        stats['best_epoch'] = stream.epoch
        stats['metrics'] = metrics
        save_checkpoint(False)
        for k, v in predictions.items():
            np.save(output / f'p_{k}.npy', v)

    elif progress.fail:
        break

    # %%
def save_prediction(split,preds, output):
    labels = Y[split]
    fortnrs = D.val['case_ids']['data'] if split == 'val' else D.test['case_ids']['data']
    assert len(preds) == len(labels)
    assert len(preds) == len(fortnrs)
    data = []
    for i in range(len(preds)):
        data.append({'case_id': fortnrs[i],
                     'pred_score': torch.tensor([preds[i]]),
                     'gt_label': torch.tensor([labels[i]])})
    with open(output / f'{split}.pkl', 'wb') as file:
        pkl.dump(data, file)


# %%
print('\nRunning the final evaluation...')
model.load_state_dict(torch.load(checkpoint_path)['model'])
stats['metrics'], predictions = evaluate(['val','test'])
for k, v in predictions.items():
    save_prediction(k, v, output)
stats['time'] = format_seconds(timer())
save_checkpoint(True)
print('Done!')
