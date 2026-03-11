# %%
import math

import logging
logging.getLogger().setLevel(logging.DEBUG)
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
import pickle as pkl
from util import *
from deep import *
from env import *
from data import Dataset, to_tensors
from metrics import calculate_metrics, make_summary

# %%
class Tokenizer(nn.Module):
    category_offsets: ty.Optional[Tensor]

    def __init__(
        self,
        d_numerical: int,
        categories: ty.Optional[ty.List[int]],
        d_token: int,
        bias: bool,
    ) -> None:
        super().__init__()
        if categories is None:
            d_bias = d_numerical
            self.category_offsets = None
            self.category_embeddings = None
        else:
            d_bias = d_numerical + len(categories)
            category_offsets = torch.tensor([0] + categories[:-1]).cumsum(0)
            self.register_buffer('category_offsets', category_offsets)
            self.category_embeddings = nn.Embedding(sum(categories), d_token)
            nn_init.kaiming_uniform_(self.category_embeddings.weight, a=math.sqrt(5))
            print(f'{self.category_embeddings.weight.shape=}')

        # take [CLS] token into account
        self.weight = nn.Parameter(Tensor(d_numerical + 1, d_token))
        self.bias = nn.Parameter(Tensor(d_bias, d_token)) if bias else None
        # The initialization is inspired by nn.Linear
        nn_init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            nn_init.kaiming_uniform_(self.bias, a=math.sqrt(5))

    @property
    def n_tokens(self) -> int:
        return len(self.weight) + (
            0 if self.category_offsets is None else len(self.category_offsets)
        )

    def forward(self, x_num: Tensor, x_cat: ty.Optional[Tensor]) -> Tensor:
        x_some = x_num if x_cat is None else x_cat
        assert x_some is not None
        x_num = torch.cat(
            [torch.ones(len(x_some), 1, device=x_some.device)]  # [CLS]
            + ([] if x_num is None else [x_num]),
            dim=1,
        )
        x = self.weight[None] * x_num[:, :, None]
        if x_cat is not None:
            x = torch.cat(
                [x, self.category_embeddings(x_cat + self.category_offsets[None])],
                dim=1,
            )
        if self.bias is not None:
            bias = torch.cat(
                [
                    torch.zeros(1, self.bias.shape[1], device=x.device),
                    self.bias,
                ]
            )
            x = x + bias[None]
        return x


class MultiheadAttention(nn.Module):
    def __init__(
        self, d: int, n_heads: int, dropout: float, initialization: str,
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
        self.attend = nn.Softmax(dim=-1)

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
        )

    def _scaled_dot_product_attention(self,query, key, value,attn_mask=None):
        scale_factor = 1 / math.sqrt(query.size(-1))
        attn_weight = query @ key.transpose(-2, -1) * scale_factor

        attn_bias = torch.zeros_like(attn_weight, dtype=query.dtype, device=query.device)
        if attn_mask is not None:
            assert attn_mask.dtype == torch.bool
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))

        attn_weight += attn_bias
        attn_weight = self.attend(attn_weight)
        if self.dropout is not None:
            attn_weight = self.dropout(attn_weight)
        return attn_weight @ value

    def forward(
        self,
        x_q: Tensor,
        x_kv: Tensor,
        mask: Tensor,
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
        d_head_value = v.shape[-1] // self.n_heads
        n_q_tokens = q.shape[1]

        q = self._reshape(q)
        k = self._reshape(k)
        v = self._reshape(v)

        if mask is not None:
            assert len(mask.shape)==2
            n_kv = mask.shape[1]
            mask = mask.repeat(1, self.n_heads*n_q_tokens).reshape(-1,self.n_heads, n_q_tokens,n_kv) # (batch_size, self.n_heads,n_q_tokens, n_kv_tokens)
            mask = mask.logical_not()

        x = self._scaled_dot_product_attention(q,k,v,attn_mask=mask)

        x = (
            x.transpose(1, 2)
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
        # tokenizer
        d_numerical: int,
        categories: ty.Optional[ty.List[int]],
        token_bias: bool,
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
        # linformer
        kv_compression: ty.Optional[float],
        kv_compression_sharing: ty.Optional[str],
        #
        d_out: int,
        mask: bool,
    ) -> None:
        assert (kv_compression is None) ^ (kv_compression_sharing is not None)

        super().__init__()
        self.mask = mask
        print("Mask attention", self.mask)

        self.tokenizer = Tokenizer(d_numerical, categories, d_token, token_bias)
        n_tokens = self.tokenizer.n_tokens

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
                        d_token, n_heads, attention_dropout, initialization,
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

    def forward(self, x_num: Tensor, x_cat: ty.Optional[Tensor]) -> Tensor:
        x = self.tokenizer(x_num, x_cat).float()
        # print(self.tokenizer.weight)

        if self.mask and self.training:
            spar = 0.15 * np.random.uniform()
            rand_mask = torch.from_numpy((np.random.random_sample(x.shape[:2]) < spar)).to(device)
            # print('spar, rand mask',spar, rand_mask)
        else:
            rand_mask = None

        for layer_idx, layer in enumerate(self.layers):
            is_last_layer = layer_idx + 1 == len(self.layers)
            layer = ty.cast(ty.Dict[str, nn.Module], layer)

            x_residual = self._start_residual(x, layer, 0)
            x_residual = layer['attention'](
                # for the last attention, it is enough to process only [CLS]
                (x_residual[:, :1] if is_last_layer else x_residual),
                x_residual,
                rand_mask,
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


# %%
if __name__ == "__main__":
    args, output = load_config()
    args['model'].setdefault('token_bias', True)
    print("token bias",args['model']['token_bias'])
    args['model'].setdefault('kv_compression', None) # TODO delete if not used
    args['model'].setdefault('kv_compression_sharing', None)# TODO delete if not used

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
    D.filter_train_nan()
    y_idx = D.info['task_idx']
    Y = {'test': D.test['Y']['data'][:, y_idx],
         'val': D.val['Y']['data'][:, y_idx],
         'train': D.train['Y']['data'][:, y_idx]}
    Y = to_tensors(Y)
    n_classes = len(set(Y[TRAIN].tolist())) if D.is_multiclass else 1
    y_info = {'endpoint': args['endpoint'], 'task_type': D.info['task_type'], 'task_idx': y_idx, 'n_classes': n_classes}
    print(y_info)
    print(D.info['C'])
    # construct X=(N,C)
    X_num = {'test': D.test['N']['data'], 'val': D.val['N']['data'], 'train': D.train['N']['data']} if D.test['N'][
                                                                                                           'data'] is not None else None
    X_cat = {'test': D.test['C']['data'], 'val': D.val['C']['data'], 'train': D.train['C']['data']} if D.test['C'][
                                                                                                           'data'] is not None else None
    X_raw = (X_num, X_cat)
    X = tuple(None if x is None else to_tensors(x) for x in X_raw)

    # locate data to cuda device if gpu is available
    device = get_device()
    if device.type != 'cpu':
        X = tuple(
            None if x is None else {k: v.to(device) for k, v in x.items()} for x in X
        )
        Y_device = {k: v.to(device) for k, v in Y.items()}
    else:
        Y_device = Y
    X_num, X_cat = X
    del X
    X_cat = {k: v.int() for k, v in X_cat.items()} if X_cat is not None else None
    if not D.is_multiclass:
        Y_device = {k: v.float() for k, v in Y_device.items()}

    model = Transformer(
        d_numerical=0 if X_num is None else X_num['train'].shape[1],
        categories=get_categories(X_cat),
        d_out=n_classes,
        **args['model'],
    ).to(device)
    checkpoint_path = output / 'checkpoint.pt'

    if args['training']['pretrain_ckpt']:
        pretrain_ckpt = f"/home/avesta/daqu/Projects/GEX/code/ClinTab-DL/output/gex_midtrain/cancerpathway_Gradcam_over_noise_repeat10_nkbcPlus1_sw_norm/" \
                        f"{args['endpoint'].replace(' ', '@')}/transformer/{args['repeat']}/{args['kfold']}/checkpoint.pt"
        model.load_state_dict(torch.load(pretrain_ckpt)['model'])
        print(f"Loaded pretrained weights from {pretrain_ckpt}")
    else:
        print("Training from scratch!")


    train_size = D.size(TRAIN)
    batch_size = args['training']['batch_size']
    epoch_size = stats['epoch_size'] = math.ceil(train_size / batch_size)
    eval_batch_size = args['training']['eval_batch_size']
    chunk_size = None

    loss_fn = (
        F.binary_cross_entropy_with_logits
        if D.is_binclass
        else F.cross_entropy
        if D.is_multiclass
        else F.mse_loss
    )

    if torch.cuda.device_count() > 1:  # type: ignore[code]
        print('Using nn.DataParallel')
        model = nn.DataParallel(model)
    stats['n_parameters'] = get_n_parameters(model)

    def needs_wd(name):
        return all(x not in name for x in ['tokenizer', '.norm', '.bias'])

    for x in ['tokenizer', '.norm', '.bias']:
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

    def apply_model(part, idx):
        return model(
            None if X_num is None else X_num[part][idx],
            None if X_cat is None else X_cat[part][idx],
        )

    @torch.no_grad()
    def evaluate(parts):
        global eval_batch_size
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
                    stats['eval_batch_size'] = eval_batch_size
                else:
                    break
            if not eval_batch_size:
                RuntimeError('Not enough memory even for eval_batch_size=1')

            # calculate pformance metrics
            nan_mask = np.isnan(Y_device[part].cpu().numpy())
            Y_masked = Y_device[part][~nan_mask]
            predictions_masked = predictions[part][~nan_mask]

            loss = loss_fn(predictions_masked, Y_masked)
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
            loss, new_chunk_size = train_with_auto_virtual_batch(
                optimizer,
                loss_fn,
                lambda x: (apply_model(TRAIN, x), Y_device[TRAIN][x]),
                batch_idx,
                chunk_size or batch_size,
            )
            epoch_losses.append(loss.detach())
            if new_chunk_size and new_chunk_size < (chunk_size or batch_size):
                stats['chunk_size'] = chunk_size = new_chunk_size
                print('New chunk size:', chunk_size)
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
        labels=Y[split]
        fortnrs=D.val['case_ids']['data'] if split == 'val' else D.test['case_ids']['data']
        assert len(preds)==len(labels)
        assert len(preds)==len(fortnrs)
        data=[]
        for i in range(len(preds)):
            data.append({'case_id':fortnrs[i],
                         'pred_score': torch.tensor([preds[i]]),
                         'gt_label': torch.tensor([labels[i]])})
        with open(output / f'{split}.pkl', 'wb') as file:
            pkl.dump(data,file)


    print('\nRunning the final evaluation...')
    model.load_state_dict(torch.load(checkpoint_path)['model'])
    stats['metrics'], predictions = evaluate(['val', 'test'])
    for k, v in predictions.items():
        save_prediction(k, v, output)
    stats['time'] = format_seconds(timer())
    save_checkpoint(True)
    print('Done!')
