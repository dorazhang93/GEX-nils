import os
import math
from torch import optim
from models import BaseVAE
import pytorch_lightning as pl
import json
import torch
from torch import Tensor

class VAEXperiment(pl.LightningModule):

    def __init__(self,
                 vae_model: BaseVAE,
                 params: dict) -> None:
        super(VAEXperiment, self).__init__()

        self.model = torch.compile(vae_model)
        self.params = params
        self.metrics = self.metrics_setup()
        self.validation_step_outputs = []
        # self.automatic_optimization = False

    def metrics_setup(self):
        inf=math.inf
        return {'val': {'loss': inf}}

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx):
        # opt = self.optimizers()

        inputs = batch
        results = self.forward(inputs)
        train_loss = self.model.loss_function(results)
        loss = train_loss['loss']

        self.log_dict({key: val.item() for key, val in train_loss.items()}, sync_dist=True)

        # opt.zero_grad()
        # self.manual_backward(loss)
        # self.clip_gradients(opt, gradient_clip_val=0.5, gradient_clip_algorithm="norm")
        # opt.step()

        # update lr every end of the epoch
        if self.trainer.is_last_batch:
            self.lr_schedulers().step()
        return loss

    # def on_after_backward(self):
    #     """
    #     Called after loss.backward() and before optimizers.step().
    #     Useful for inspecting gradients.
    #     """
    #     if self.trainer.global_step % 1 == 0:  # Log periodically to avoid huge files
    #         for name, param in self.named_parameters():
    #             if param.grad is not None:
    #                 # Log the gradient histogram
    #                 print(name,param.grad)
    #                 # You can also log the gradient norm
    #                 print(name, param.grad.norm())
    #                 import sys
    #                 sys.exit()
    #
    #             else:
    #                 print(f"Gradient for {name} is None!")  # Useful for debugging why a grad might be missing


    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        inputs = batch

        results = self.forward(inputs)

        val_loss = self.model.loss_function(results)

        self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True)
        self.validation_step_outputs.append(val_loss)
        return val_loss

    def on_validation_epoch_end(self,):
        outputs = self.validation_step_outputs
        val_loss={'loss': 0}
        steps=0
        for output in outputs:
            steps+=1
            val_loss['loss']+=output['loss'].cpu().numpy()
        val_loss={key: val/steps for key,val in val_loss.items()}
        print("steps %d" % steps, val_loss)
        if val_loss['loss']< self.metrics['val']['loss']:
            self.metrics['val']=val_loss
            with open(os.path.join(self.logger.log_dir,"metrics.json"),"w") as outfile:
                outfile.write(json.dumps(self.metrics))
        self.validation_step_outputs.clear()

    def configure_optimizers(self):

        optims = []
        scheds = []

        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.params['LR'],
                               weight_decay=self.params['weight_decay'])
        optims.append(optimizer)
        # Check if more than 1 optimizer is required (Used for adversarial training)
        try:
            if self.params['LR_2'] is not None:
                optimizer2 = optim.Adam(getattr(self.model,self.params['submodel']).parameters(),
                                        lr=self.params['LR_2'])
                optims.append(optimizer2)
        except:
            pass

        try:
            if self.params['scheduler_gamma'] is not None:
                scheduler = optim.lr_scheduler.ExponentialLR(optims[0],
                        gamma = self.params['scheduler_gamma'])
                scheds.append(scheduler)

                # Check if another scheduler is required for the second optimizer
                try:
                    if self.params['scheduler_gamma_2'] is not None:
                        scheduler2 = optim.lr_scheduler.ExponentialLR(optims[1],
                                gamma = self.params['scheduler_gamma_2'])
                        scheds.append(scheduler2)
                except:
                    pass
                return optims, scheds
        except:
            return optims
