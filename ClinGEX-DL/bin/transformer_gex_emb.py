# %%
import math
import logging

logging.getLogger().setLevel(logging.DEBUG)
# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
import pickle as pkl
from util import *
from deep import *
from env import *
from torch.utils.data import Dataset, DataLoader
from metrics import calculate_metrics, make_summary
import pickle
import numpy as np
# %%

class MultiheadAttention(nn.Module):
    def __init__(
        self, d: int, n_heads: int, dropout: float, initialization: str, is_LSA: bool
    ) -> None:
        if n_heads > 1:
            assert d % n_heads == 0
        assert initialization in ['xavier', 'kaiming']

        super().__init__()
        self.W_q = nn.Linear(d, d)
        self.W_k = nn.Linear(d, d)
        self.W_v = nn.Linear(d, d)
        self.W_out = nn.Linear(d, d) if n_heads > 1 else None
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout) if dropout else None
        self.is_LSA = is_LSA
        self.attend = nn.Softmax(dim=-1)
        if self.is_LSA:
            print("is_LSA True!!!")

        for m in [self.W_q, self.W_k, self.W_v]:
            if initialization == 'xavier' and (n_heads > 1 or m is not self.W_v):
                # gain is needed since W_qkv is represented with 3 separate layers
                nn_init.xavier_uniform_(m.weight, gain=1 / math.sqrt(2))
            nn_init.zeros_(m.bias)
        if self.W_out is not None:
            nn_init.zeros_(self.W_out.bias)


    def _reshape(self, x: Tensor) -> Tensor:
        batch_size, n_tokens, d = x.shape
        d_head = d // self.n_heads
        return (
            x.reshape(batch_size, n_tokens, self.n_heads, d_head)
            .transpose(1, 2)
            .reshape(batch_size * self.n_heads, n_tokens, d_head)
        )

    def forward(
        self,
        x_q: Tensor,
        x_kv: Tensor,
        key_compression: ty.Optional[nn.Linear],
        value_compression: ty.Optional[nn.Linear],
    ) -> Tensor:
        q, k, v = self.W_q(x_q), self.W_k(x_kv), self.W_v(x_kv)
        for tensor in [q, k, v]:
            assert tensor.shape[-1] % self.n_heads == 0
        if key_compression is not None:
            assert value_compression is not None
            k = key_compression(k.transpose(1, 2)).transpose(1, 2)
            v = value_compression(v.transpose(1, 2)).transpose(1, 2)
        else:
            assert value_compression is None

        batch_size = len(q)
        d_head_key = k.shape[-1] // self.n_heads
        d_head_value = v.shape[-1] // self.n_heads
        n_q_tokens = q.shape[1]

        q = self._reshape(q)
        k = self._reshape(k)
        dots = q @ k.transpose(1, 2) / math.sqrt(d_head_key) # shape: (batch_size * self.n_heads, n_q_tokens, n_tokens)

        if self.is_LSA:
            mask = torch.eye(n_q_tokens, n_q_tokens)
            mask = torch.nonzero((mask == 1), as_tuple=False)
            dots[:, mask[:,0], mask[:,1]] = -987654321


        attention = self.attend(dots)
        if self.dropout is not None:
            attention = self.dropout(attention)
        x = attention @ self._reshape(v)
        x = (
            x.reshape(batch_size, self.n_heads, n_q_tokens, d_head_value)
            .transpose(1, 2)
            .reshape(batch_size, n_q_tokens, self.n_heads * d_head_value)
        )
        if self.W_out is not None:
            x = self.W_out(x)
        return x


class Transformer(nn.Module):
    """Transformer.

    References:
    - https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html
    - https://github.com/facebookresearch/pytext/tree/master/pytext/models/representations/transformer
    - https://github.com/pytorch/fairseq/blob/1bba712622b8ae4efb3eb793a8a40da386fe11d0/examples/linformer/linformer_src/modules/multihead_linear_attention.py#L19
    """

    def __init__(
        self,
        *,
        n_tokens: int,
        # transformer
        n_layers: int,
        d_token: int,
        n_heads: int,
        d_ffn_factor: float,
        attention_dropout: float,
        ffn_dropout: float,
        residual_dropout: float,
        activation: str,
        prenormalization: bool,
        initialization: str,
        is_LSA: bool,
        # linformer
        kv_compression: ty.Optional[float],
        kv_compression_sharing: ty.Optional[str],
        #
        d_out: int,
    ) -> None:
        assert (kv_compression is None) ^ (kv_compression_sharing is not None)

        super().__init__()
        self.cls_weight = nn.Parameter(Tensor(1, d_token))
        nn_init.kaiming_uniform_(self.cls_weight, a=math.sqrt(5))
        n_tokens = n_tokens +1

        def make_kv_compression():
            assert kv_compression
            compression = nn.Linear(
                n_tokens, int(n_tokens * kv_compression), bias=False
            )
            if initialization == 'xavier':
                nn_init.xavier_uniform_(compression.weight)
            return compression

        self.shared_kv_compression = (
            make_kv_compression()
            if kv_compression and kv_compression_sharing == 'layerwise'
            else None
        )

        def make_normalization():
            return nn.LayerNorm(d_token)

        d_hidden = int(d_token * d_ffn_factor)
        self.layers = nn.ModuleList([])
        for layer_idx in range(n_layers):
            layer = nn.ModuleDict(
                {
                    'attention': MultiheadAttention(
                        d_token, n_heads, attention_dropout, initialization, is_LSA
                    ),
                    'linear0': nn.Linear(
                        d_token, d_hidden * (2 if activation.endswith('glu') else 1)
                    ),
                    'linear1': nn.Linear(d_hidden, d_token),
                    'norm1': make_normalization(),
                }
            )
            if not prenormalization or layer_idx:
                layer['norm0'] = make_normalization()
            if kv_compression and self.shared_kv_compression is None:
                layer['key_compression'] = make_kv_compression()
                if kv_compression_sharing == 'headwise':
                    layer['value_compression'] = make_kv_compression()
                else:
                    assert kv_compression_sharing == 'key-value'
            self.layers.append(layer)

        self.activation = get_activation_fn(activation)
        self.last_activation = get_nonglu_activation_fn(activation)
        self.prenormalization = prenormalization
        self.last_normalization = make_normalization() if prenormalization else None
        self.ffn_dropout = ffn_dropout
        self.residual_dropout = residual_dropout
        self.head = nn.Linear(d_token, d_out)

    def _get_kv_compressions(self, layer):
        return (
            (self.shared_kv_compression, self.shared_kv_compression)
            if self.shared_kv_compression is not None
            else (layer['key_compression'], layer['value_compression'])
            if 'key_compression' in layer and 'value_compression' in layer
            else (layer['key_compression'], layer['key_compression'])
            if 'key_compression' in layer
            else (None, None)
        )

    def _start_residual(self, x, layer, norm_idx):
        x_residual = x
        if self.prenormalization:
            norm_key = f'norm{norm_idx}'
            if norm_key in layer:
                x_residual = layer[norm_key](x_residual)
        return x_residual

    def _end_residual(self, x, x_residual, layer, norm_idx):
        if self.residual_dropout:
            x_residual = F.dropout(x_residual, self.residual_dropout, self.training)
        x = x + x_residual
        if not self.prenormalization:
            x = layer[f'norm{norm_idx}'](x)
        return x

    def forward(self, x_num: Tensor) -> Tensor:
        x_cls = self.cls_weight[None]* (torch.ones(len(x_num),1,device=x_num.device))[:,:,None]

        x = torch.cat([x_cls,
                       x_num],dim=1)


        for layer_idx, layer in enumerate(self.layers):
            is_last_layer = layer_idx + 1 == len(self.layers)
            layer = ty.cast(ty.Dict[str, nn.Module], layer)

            x_residual = self._start_residual(x, layer, 0)
            x_residual = layer['attention'](
                # for the last attention, it is enough to process only [CLS]
                (x_residual[:, :1] if is_last_layer else x_residual),
                x_residual,
                *self._get_kv_compressions(layer),
            )
            if is_last_layer:
                x = x[:, : x_residual.shape[1]]
            x = self._end_residual(x, x_residual, layer, 0)

            x_residual = self._start_residual(x, layer, 1)
            x_residual = layer['linear0'](x_residual)
            x_residual = self.activation(x_residual)
            if self.ffn_dropout:
                x_residual = F.dropout(x_residual, self.ffn_dropout, self.training)
            x_residual = layer['linear1'](x_residual)
            x = self._end_residual(x, x_residual, layer, 1)

        assert x.shape[1] == 1
        x = x[:, 0]
        if self.last_normalization is not None:
            x = self.last_normalization(x)
        feature = self.last_activation(x)
        x = self.head(feature)
        x = x.squeeze(-1)
        return x

class GEX(Dataset):
    def __init__(self,
                 data_dir: str,
                 split: str,
                 endpoint: str,
                 **kwargs):
        self.endpoint = endpoint
        self.data_list = self._load_from_pkl(f"{data_dir}/meta.pickle",split)
        if split=='train':
            self.data_list = self._remove_gt_nan()
        self.n_sample = self.__len__()
        print(f"loaded {len(self.data_list)} samples for endpoint-{self.endpoint}")

    def _load_from_pkl(self, data_path, split):
        data_dict = pickle.load(open(data_path,"rb"))
        return data_dict[split]
    def _remove_gt_nan(self):
        data_list_new=[]
        for sample in self.data_list:
            if sample['gt_labels'][self.endpoint] == sample['gt_labels'][self.endpoint]:
                data_list_new.append(sample)
        return data_list_new
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        sample=self.data_list[idx]
        emb_file = sample['gex_emb_file']
        gt_label = sample['gt_labels'][self.endpoint]
        gex_emb = np.load(emb_file)['data']

        sid = os.path.basename(emb_file)[:-4]
        return torch.from_numpy(gex_emb), torch.from_numpy(np.array(gt_label,dtype=float)), sid

    @property
    def is_binclass(self) -> bool:
        binary_classify_tasks = ['Multifocality', 'LNM status', 'SLNM status', '5-year survival status',
                                 '5-year RF status', '5-year DRF status']
        return self.endpoint in binary_classify_tasks
    @property
    def task_type(self):
        return 'binaryClass' if self.is_binclass else 'regression'

# %%
if __name__ == "__main__":
    args, output = load_config()
    args['model'].setdefault('kv_compression', None) # TODO delete if not used
    args['model'].setdefault('kv_compression_sharing', None)# TODO delete if not used

    # %%
    zero.set_randomness(args['repeat'])
    dataset_dir = str(get_data_path(args['dataset']) / str(args["repeat"]) / str(args["kfold"]))
    stats: ty.Dict[str, ty.Any] = {
        'dataset': dataset_dir,
        'algorithm': Path(__file__).stem,
        **load_json(output / 'stats.json'),
    }
    timer = zero.Timer()
    timer.run()

    # build datasets loader
    train_dataset = GEX(data_dir=dataset_dir,split='train',endpoint=args['endpoint'])
    val_dataset = GEX(data_dir=dataset_dir, split='val', endpoint=args['endpoint'])
    test_dataset = GEX(data_dir=dataset_dir, split='test',endpoint=args['endpoint'])

    train_size = train_dataset.n_sample
    batch_size = args['training']['batch_size']
    epoch_size = stats['epoch_size'] = math.ceil(train_size / batch_size)
    eval_batch_size = batch_size*2
    chunk_size = None

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=batch_size,
        num_workers=8,
    )
    val_dataloader = DataLoader(val_dataset,
                                shuffle=False,
                                batch_size=eval_batch_size,
                                num_workers=8,
                                )
    test_dataloader = DataLoader(test_dataset,
                                shuffle=False,
                                batch_size=eval_batch_size,
                                num_workers=8,)

    eval_data={'val':val_dataloader,
               'test':test_dataloader}

    device = get_device()
    model = Transformer(
        d_out=1,
        **args['model'],
    ).to(device)
    checkpoint_path = output / 'checkpoint.pt'

    print("Training from scratch!")


    loss_fn = (
        F.binary_cross_entropy_with_logits
        if train_dataset.is_binclass
        else F.mse_loss
    )

    if torch.cuda.device_count() > 1:  # type: ignore[code]
        print('Using nn.DataParallel')
        model = nn.DataParallel(model)
    stats['n_parameters'] = get_n_parameters(model)

    def needs_wd(name):
        return all(x not in name for x in ['.norm', '.bias'])

    for x in ['.norm', '.bias']:
        assert any(x in a for a in (b[0] for b in model.named_parameters()))
    parameters_with_wd = [v for k, v in model.named_parameters() if needs_wd(k)]
    parameters_without_wd = [v for k, v in model.named_parameters() if not needs_wd(k)]

    optimizer = make_optimizer(
        args['training']['optimizer'],
        (
            [
                {'params': parameters_with_wd},
                {'params': parameters_without_wd, 'weight_decay': 0.0},
            ]
        ),
        args['training']['lr'],
        args['training']['weight_decay'],
    )

    # scheduler, _, _ = lib.make_lr_schedule(optimizer,args['lr_schedule'])
    stream = zero.Stream(IndexLoader(train_size, batch_size, True, device))
    progress = zero.ProgressTracker(args['training']['patience'])
    training_log = {TRAIN: [], VAL: [], TEST: []}
    timer = zero.Timer()

    def print_epoch_info():
        print(f'\n>>> Epoch {stream.epoch} | {format_seconds(timer())} | {output}')
        print(
            ' | '.join(
                f'{k} = {v}'
                for k, v in {
                    'lr': get_lr(optimizer),
                    'batch_size': batch_size,
                    'chunk_size': chunk_size,
                    'epoch_size': stats['epoch_size'],
                    'n_parameters': stats['n_parameters'],
                }.items()
            )
        )


    @torch.no_grad()
    def evaluate(parts):
        global eval_batch_size
        model.eval()
        metrics = {}
        predictions = {}
        gt_labels={}
        case_ids={}
        for part in parts:
            predictions_list=[]
            gt_labels_list=[]
            case_ids_list=[]
            for batch in eval_data[part]:
                predictions_list.append(model(batch[0].to(device)))
                gt_labels_list.append(batch[1])
                case_ids_list.append(batch[2])
            predictions[part]=torch.cat(predictions_list)
            gt_labels[part]=torch.cat(gt_labels_list)
            case_ids[part]=case_ids_list

            # calculate pformance metrics
            nan_mask = np.isnan(gt_labels[part].numpy())
            Y_masked = gt_labels[part][~nan_mask]
            predictions_masked = predictions[part][~nan_mask]

            loss = loss_fn(predictions_masked, Y_masked.to(device))
            predictions[part] = predictions[part].cpu()
            metrics[part] = calculate_metrics(
                train_dataset.task_type,
                Y_masked.numpy(),  # type: ignore[code]
                predictions_masked.cpu().numpy(),  # type: ignore[code]
                'logits',
            )
            metrics[part]['loss'] = loss.cpu().numpy().item()

        for part, part_metrics in metrics.items():
            print(f'[{part:<5}]', make_summary(part_metrics))
        return metrics, predictions, gt_labels, case_ids

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
        for step_idx, batch in enumerate(train_dataloader):
            batch_gex, batch_gt, _ = batch

            optimizer.zero_grad()
            loss = loss_fn(model(batch_gex.to(device)),batch_gt.to(device))
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.detach())
        # scheduler.step()
        epoch_losses = torch.stack(epoch_losses).tolist()
        training_log[TRAIN].extend(epoch_losses)
        print(f'[{TRAIN}] loss = {round(sum(epoch_losses) / len(epoch_losses), 3)}')

        metrics, predictions, gt_labels, _ = evaluate([VAL])
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
    def save_prediction(split,preds,labels, case_ids, output):
        case_ids = [x for x_set in case_ids for x in x_set]
        assert len(preds)==len(labels)
        assert len(preds)==len(case_ids)
        data=[]
        for i in range(len(preds)):
            data.append({'case_id':case_ids[i],
                         'pred_score': torch.tensor([preds[i]]),
                         'gt_label': torch.tensor([labels[i]])})
        with open(output / f'{split}.pkl', 'wb') as file:
            pkl.dump(data,file)


    print('\nRunning the final evaluation...')
    model.load_state_dict(torch.load(checkpoint_path)['model'])
    stats['metrics'], predictions, gt_labels, case_ids = evaluate(['val', 'test'])
    for k, v in predictions.items():
        save_prediction(k, v, gt_labels[k],case_ids[k], output)
    stats['time'] = format_seconds(timer())
    save_checkpoint(True)
    print('Done!')
