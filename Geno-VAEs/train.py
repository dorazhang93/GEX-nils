from config import *
from models import *
from experiment import VAEXperiment
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from dataloader import VAEDataset
import os, shutil
os.environ['CUDA_VISIBLE_DEVICES']='0,1'

config, out_dir, config_name = load_config()
if os.path.isdir(out_dir):
    shutil.rmtree(out_dir)
save_dir = config['logging_params']['save_dir'] + config["data_params"]["data_name"]
tb_logger =  TensorBoardLogger(save_dir=save_dir,
                               name=config['model_params']['name'], version=config_name, 
                               log_graph=False)

# For reproducibility
torch.manual_seed(config['exp_params']['manual_seed'])
gpus = config['trainer_params']['gpus']

model = vae_models[config['model_params']['name']](**config['model_params'])

experiment = VAEXperiment(model,
                          config['exp_params'])


data = VAEDataset(**config["data_params"], pin_memory=len(gpus) != 0)

runner = Trainer(logger=tb_logger,
                 callbacks=[
                     LearningRateMonitor(),
                     ModelCheckpoint(save_top_k=1,
                                     dirpath =os.path.join(tb_logger.log_dir , "checkpoints"),
                                     monitor= "val_loss",
                                     save_last= True),
                 ],
                 gradient_clip_val=0.5,
                 accumulate_grad_batches=4,
                 precision='16-mixed',
                 accelerator="gpu",
                 devices=gpus,
                 max_epochs=config['trainer_params']['max_epochs'],
                 configs_dict=config)


Path(tb_logger.log_dir).mkdir(exist_ok=True, parents=True)

print(f"======= Training {config['model_params']['name']} =======")
runner.fit(experiment, datamodule=data)

# In pl.LightningModule, data.setup("fit") will be automatically called in runner.fit()
# there are four states of trainerFn (defined in pytorch_lightning.trainer.states)
#     FITTING = "fit"
#     VALIDATING = "validate"
#     TESTING = "test"
#     PREDICTING = "predict"
# - ``TrainerFn.FITTING`` - ``RunningStage.{SANITY_CHECKING, TRAINING, VALIDATING}``
# - ``TrainerFn.VALIDATING`` - ``RunningStage.VALIDATING``
# - ``TrainerFn.TESTING`` - ``RunningStage.TESTING``
# - ``TrainerFn.PREDICTING`` - ``RunningStage.PREDICTING``