import os 

from pytorch_lightning import Callback

import yaml
import argparse
import numpy as np 
import pdb
from library import models2
#from library.models2 import helpers
#from library.models2.helpers import vae_models, vae_architectures, #
from library.models2.helpers import *
from library.architectures import Discriminator 
from experiment import RlExperiment
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import MLFlowLogger

import confuse


torch.set_default_dtype(torch.float64)

parser = argparse.ArgumentParser(description='Generic runner for Representation Learning with VAE')
parser.add_argument('--config', '-c',
                    dest='filename',
                    metavar='FILE',
                    help='path to config file',
                    default='configs/ADIDAS/vae.yaml')
parser.add_argument('--experiment_name',
                    type=str, default='VaeExperiment',
                    metavar='N', help='specifies the experiment name for better tracking later')
parser.add_argument('--run_name',
                    type=str, default='VaeGuassian',
                    metavar='N', help='specifies the subfolder name where visualization shall be stored')
# model/architecture params
parser.add_argument('--img_encoder', type=str, default=None,
                    help='specifies the encoder for image data (default: config file)')
parser.add_argument('--img_decoder', type=str, default=None,
                    help='specifies the decoder for image data (default: config file)')
parser.add_argument('--attr_encoder', type=str, default=None,
                    help='specifies the encoder for attribute data (default: config file')
parser.add_argument('--attr_decoder', type=str, default=None,
                    help='specifies the decoder for attribute data (default: config file')
parser.add_argument('--latent_dim', type=int, default=None,
                    help='latent dim for VAE (default: config file)')
# experiment params
parser.add_argument('--batch_size', type=int, default=None,
                    help='size of batch (default: config file)')
parser.add_argument('--learning_rate', type=float, default=None,
                    help='learning rate for training (default: config file)')
parser.add_argument('--weight_decay', type=float, default=None,
                    help='weight_decay for learning rate scheduler (default: config file)')
parser.add_argument('--scheduler_gamma', type=float, default=None,
                    help='scheduler gamma for learning rate scheduler (default: config file)')
# trainer params
parser.add_argument('--gpus', type=int, default=None,
                    help='number of gpus available (default: config file)')
parser.add_argument('--max_epochs', type=int, default=None,
                    help='number of epochs (default: config file)')

# model params
# GaussianVae
#parser.add_argument()

# BetaVae 
#parser.add_argument()

# InfoVae   
parser.add_argument('--beta', type=float, default=None, metavar='N',
                    help='')
parser.add_argument('--reg_weight', type=int, default=None, metavar='N',
                    help='')
parser.add_argument('--kernel_type', type=str, default=None, metavar='N',
                    help='')
parser.add_argument('--latent_var', type=float, default=None, metavar='N',
                    help='')

# CatVae
parser.add_argument('--temperature', type=float, default=None, metavar='N',
                    help='sets the temperature')
parser.add_argument('--anneal_rate', type=float, default=None, metavar='N',
                    help='')
parser.add_argument('--anneal_interval', type=int, metavar='N',
                    help='')
parser.add_argument('--alpha', type=float, default=None, metavar='N',
                    help='')
parser.add_argument('--probability', type=bool, default=None, metavar='N',
                    help='')

# GaussianMixVae 
parser.add_argument('--temperature_bound', type=int, default=None, metavar='N',
                    help='')
parser.add_argument('--decrease_temp', type=bool, default=None, metavar='N',
                    help='')
parser.add_argument('--beta1', type=int, default=None, metavar='N',
                    help='')
parser.add_argument('--beta2', type=int, default=None, metavar='N',
                    help='')

# JointVae
#parser.add_argument()
parser.add_argument('--latent_min_capacity', type=float, default=None, metavar='N',
                    help='')
parser.add_argument('--latent_max_capacity', type=float, default=None, metavar='N',
                    help='')
parser.add_argument('--latent_gamma', type=float, default=None, metavar='N',
                    help='')
parser.add_argument('--latent_num_iter', type=float, default=None, metavar='N',
                    help='')
parser.add_argument('--categorical_min_capacity', type=float, default=None, metavar='N',
                    help='')
parser.add_argument('--categorical_max_capacity', type=float, default=None, metavar='N',
                    help='')
parser.add_argument('--categorical_gamma', type=float, default=None, metavar='N',
                    help='')
parser.add_argument('--categorical_num_iter',  type=float, default=None, metavar='N',
                    help='')

# DIPVae
#parser.add_argument()
parser.add_argument('--lambda_dig', type=float, default=None, metavar='N',
                    help='')
parser.add_argument('--lambda_offdig', type=float, default=None, metavar='N',
                    help='')


args = parser.parse_args()


with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

config = update_config(config=config, args=args)






class MetricsCallback(Callback):
    """
    """
    def __init__(self):
        super().__init__()
        self.metrics = []
    
    def on_validation_end(self, trainer, pl_module):
        self.metrics.append(trainer.callback_metrics)
    






## define objective

def objective(trial, config, args):



    checkpoint_callback = pl.callbacks.ModelCheckpoint(
                                os.path.join(MODEL_DIR, f"trial_{trial.number}, {epoch}"), monitor='val_loss'
    )

    metrics_callback = MetricsCallback()
    
    model = parse_model_config(config)
    mlflow_logger = MLFlowLogger(experiment_name=args.experiment_name)

    experiment = RlExperiment(model,
                            discriminator=discriminator, 
                            params=config['exp_params'],
                            model_hyperparams=config['model_hyperparams'],
                            run_name=args.run_name,
                            experiment_name=args.experiment_name)

    trainer = Trainer(default_save_path=config['logging_params']['save_dir'],
                    logger=mlflow_logger,
                    check_val_every_n_epoch=1,
                    train_percent_check=1.,
                    val_percent_check=1.,
                    num_sanity_val_steps=0,
                    callback=[metrics_callback],
                    early_stop_callback=PytorchLightningPruningCallback(trial, monitor='val_acc'),
                    fast_dev_run=False,
                    **config['trainer_params'])
    






# create optuna study
pruner = optuna.pruners.MedianPruner() if args.pruning else optuna.pruners.NopPruner()

study = optuna.create_study(direction='maximize', pruner=pruner)
study.optimize(objective, n_trials=100, timeout=600)

print("Number of finished traisl")




## store best values!

trial = study.best_trial

for key, value in trial.params.items():
    print("  {}: {}".format(key, value))




# do not what this is doing!
shutil.rmtree(MODEL_DIR)