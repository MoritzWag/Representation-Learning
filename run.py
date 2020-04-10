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
from pytorch_lightning.logging import MLFlowLogger

parser = argparse.ArgumentParser(description='Generic runner for Representation Learning with VAE')
parser.add_argument('--config', '-c',
                    dest='filename',
                    metavar='FILE',
                    help='path to config file',
                    default='configs/vae.yaml')
parser.add_argument('--experiment_name',
                    type=str, default=None,
                    metavar='N', help='specifies the experiment name for better tracking later')

args = parser.parse_args()
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)


model = parse_model_config(config)



## build logger

mlflow_logger = MLFlowLogger(
                    experiment_name=args.experiment_name,
                    tracking_uri=None,
                    tags=None)

pdb.set_trace()
## build experiment
experiment = RlExperiment(model,
                        params=config['exp_params'])

## build trainer
runner = Trainer(min_nb_epochs=1,
                logger=mlflow_logger,
                train_percent_check=1.,
                val_percent_check=1.,
                **config['trainer_params']
                )


## run trainer

print(f"======= Training {config['model_params']['name']} ==========")
runner.fit(experiment)