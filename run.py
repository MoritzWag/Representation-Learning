import yaml
import argparse
import numpy as np 
import pdb
from library import models2
#from library.models2 import helpers
#from library.models2.helpers import vae_models, vae_architectures, #
from library.models2.helpers import *
from experiment import RlExperiment
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import MLFlowLogger

parser = argparse.ArgumentParser(description='Generic runner for Representation Learning with VAE')
parser.add_argument('--config', '-c',
                    dest='filename',
                    metavar='FILE',
                    help='path to config file',
                    default='configs/vae.yaml')
parser.add_argument('--experiment_name',
                    type=str, default='VaeExperiment',
                    metavar='N', help='specifies the experiment name for better tracking later')

args = parser.parse_args()
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)


model = parse_model_config(config)


## build logger
#mlflow_logger = MLFlowLogger(
#                    experiment_name=args.experiment_name,
#                    tracking_uri=None,
#                    tags=None)

mlflow_logger = MLFlowLogger(
                    experiment_name=args.experiment_name)


## build experiment
experiment = RlExperiment(model,
                        params=config['exp_params'])


## build trainer
## do I need the default_save_path?
runner = Trainer(default_save_path=config['logging_params']['save_dir'],
                min_epochs=1,
                logger=mlflow_logger,
                check_val_every_n_epoch=1,
                train_percent_check=1.,
                val_percent_check=1.,
                num_sanity_val_steps=5,
                early_stop_callback=False,
                fast_dev_run=False,
                **config['trainer_params']
                )

## run trainer
print(f"======= Training {config['model_params']['name']} ==========")
runner.fit(experiment)
runner.test()