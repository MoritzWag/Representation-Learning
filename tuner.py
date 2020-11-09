import os 

from pytorch_lightning import Callback

import yaml
import argparse
import numpy as np 
import pdb
from library import models2

from library.models2.helpers import *
from library.architectures import Discriminator 
from experiment import RlExperiment
from pytorch_lightning import Trainer
import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger

import optuna
from optuna.integration import PyTorchLightningPruningCallback
from datetime import datetime

import plotly.io as pio 

pio.orca.config.use_xvfb = True 

torch.set_default_dtype(torch.float64)

parser = argparse.ArgumentParser(description='Generic runner for Representation Learning with VAE')
parser.add_argument('--config', '-c',
                    dest='filename',
                    metavar='FILE',
                    help='path to config file',
                    default='configs/ADIDAS/gaussmix_vae.yaml')
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
parser.add_argument('--kld_weight', type=float, default=None,
                    help='Weight for the KL-Divergence term in the ELBO of VAE Models')

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
parser.add_argument('--lambda_dig', type=float, default=None, metavar='N',
                    help='')
parser.add_argument('--lambda_offdig', type=float, default=None, metavar='N',
                    help='')


#tuner params
parser.add_argument('--n_trials', type=int, default=None, metavar='N',
                    help='specifies the number of trials for tuning')
parser.add_argument('--timeout', type=int, default=None, metavar='N',
                    help="specifies the total seconds used for tuning")
parser.add_argument('--min_resource', type=int, default=None, metavar='N',
                    help='minimum resource use for each configuration during tuning')
parser.add_argument('--reduction_factor', type=int, default=None, metavar='N',
                    help='factor by which number of configurations are reduced')


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

DIR = os.getcwd()
MODEL_DIR = os.path.join(DIR, "result")

start = datetime.now()

## define objective
def objective(trial):
    """
    """
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        os.path.join(MODEL_DIR, "trial_{}".format(trial.number), "{epoch}"), monitor="val_acc"
    )

    metrics_callback = MetricsCallback()
    
    model = parse_model_config(config, trial=trial)
    mlflow_logger = MLFlowLogger(experiment_name=args.experiment_name)

    experiment = RlExperiment(model, 
                            params=config['exp_params'],
                            log_params=config['logging_params'],
                            model_hyperparams=config['model_hyperparams'],
                            run_name=args.run_name,
                            experiment_name=args.experiment_name)

    runner = Trainer(default_save_path=config['logging_params']['save_dir'],
                    logger=mlflow_logger,
                    check_val_every_n_epoch=1,
                    train_percent_check=1,
                    val_percent_check=1,
                    num_sanity_val_steps=0,
                    callbacks=[metrics_callback],
                    early_stop_callback=PyTorchLightningPruningCallback(trial, monitor='mut_info'),
                    fast_dev_run=False,
                    **config['trainer_params'])
    

    runner.fit(experiment)

    return metrics_callback.metrics[-1]['mut_info'].item()


n_train_iter = config['trainer_params']['max_epochs']
study = optuna.create_study(
    direction='maximize',
    pruner=optuna.pruners.HyperbandPruner(
        min_resource=args.min_resource,
        max_resource=n_train_iter,
        reduction_factor=args.reduction_factor
    )
)

if args.n_trials is not None:
    print(f"tune with n_trials: {args.n_trials}")
    study.optimize(objective, n_trials=ars.n_trials)
if args.timeout is not None:
    print(f"tune with timeout: {args.timeout}")
    study.optimize(objective, timeout=args.timeout)

end = datetime.now()
diff = end - start
print(f"total tunning time was: {diff}")

# retrieve/print best trial
best_trial = study.best_trial
print(best_trial)

# retrieve/print best params
best_params = study.best_params
print(best_params)

# save results in dateframe
df = study.trials_dataframe()
df.to_csv(f"hb_{args.run_name}.csv")


fig_intermediate_values = optuna.visualization.plot_intermediate_values(study)
fig_intermediate_values.write_image(f'hb_tune_interm_{args.run_name}.png')

fig_opt_history = optuna.visualization.plot_optimization_history(study)
fig_opt_history.write_image(f'hb_tune_opt_hist_{args.run_name}.png')


fig_hyp_importance = optuna.visualization.plot_param_importances(study)
fig_hyp_importance.write_image(f"hb_tune_hyp_imp_{args.run_name}.png")
