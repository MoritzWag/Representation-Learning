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

import torch.backends.cudnn as cudnn




torch.set_default_dtype(torch.float64)

parser = argparse.ArgumentParser(description='Generic runner for Representation Learning with VAE')
parser.add_argument('--config', '-c',
                    dest='filename',
                    metavar='FILE',
                    help='path to config file',
                    default='configs/ADIDAS/beta_vae.yaml')
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

##logging params
parser.add_argument('--manual_seed', type=int, default=None,
                    help='seed for reproducibility (default: config file)')

# model params
parser.add_argument('--kld_weight', type=float, default=None,
                    help='Weight for the KL-Divergence term in the ELBO of VAE Models')
# GaussianVae
#parser.add_argument()

# BetaVae 
parser.add_argument('--beta', type=float, default=None, metavar='N',
                    help='')
parser.add_argument('--restrict_capacity', type=bool, default=None, metavar='N',
                    help='')
parser.add_argument('--capacity_max_iter', type=int, default=None, metavar='N',
                    help='')
parser.add_argument('--max_capacity', type=int, default=None, metavar='N')

# InfoVae   
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
parser.add_argument('--cat_weight', type=int, default=None, metavar='N',
                    help='')
parser.add_argument('--cont_weight', type=int, default=None, metavar='N',
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

model = parse_model_config(config, trial=None)


mlflow_logger = MLFlowLogger(
                    experiment_name=args.experiment_name)


#ddiscriminator = Discriminator(latent_dim=config['img_arch_params']['latent_dim'])

#For reproducibility
torch.manual_seed(config['logging_params']['manual_seed'])
np.random.seed(config['logging_params']['manual_seed'])
cudnn.deterministic = True 
cudnn.benchmark = False

## build experiment
experiment = RlExperiment(model,
                        params=config['exp_params'],
                        log_params=config['logging_params'],
                        model_hyperparams=config['model_hyperparams'],
                        run_name=args.run_name,
                        experiment_name=args.experiment_name)


## build trainer
## do I need the default_save_path?
runner = Trainer(default_save_path=config['logging_params']['save_dir'],
                min_epochs=1,
                logger=mlflow_logger,
                check_val_every_n_epoch=1,
                train_percent_check=1,
                val_percent_check=1,
                num_sanity_val_steps=5,
                early_stop_callback=False,
                fast_dev_run=False,
                **config['trainer_params']
                )

## run trainer
print(f"======= Training {config['model_params']['name']} ==========")
runner.fit(experiment)
runner.test()