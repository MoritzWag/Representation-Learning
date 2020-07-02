import os

experiment_name = "basic_runner"
num_epochs = 80
command1 = f"python run.py --config configs/ADIDAS/vae.yaml --latent_dim 10 --experiment_name {experiment_name} --run_name VanillaVae --max_epochs {num_epochs}"
command2 = f"python run.py --config configs/ADIDAS/info_vae.yaml --latent_dim 10 --experiment_name {experiment_name} --run_name InfoVae --max_epochs {num_epochs}"
command3 = f"python run.py --config configs/ADIDAS/beta_vae.yaml --latent_dim 10 --experiment_name {experiment_name} --run_name BetaVae --max_epochs {num_epochs}"
command4 = f"python run.py --config configs/ADIDAS/gaussmix_vae.yaml --latent_dim 10 --experiment_name {experiment_name} --run_name GaussMixVae --max_epochs {num_epochs}"
command5 = f"python run.py --config configs/ADIDAS/dip_vae.yaml --latent_dim 10 --experiment_name {experiment_name} --run_name DIPVae --max_epochs {num_epochs}"
command6 = f"python run.py --config configs/ADIDAS/autoenc.yaml --latent_dim 10 --experiment_name {experiment_name} --run_name Autoenc --max_epochs {num_epochs}"
command7 = f"python run.py --config configs/ADIDAS/pca.yaml --latent_dim 10 --experiment_name {experiment_name} --run_name pca --max_epochs 10"

os.system(command1)
os.system(command2)
os.system(command3)
os.system(command4)
os.system(command5)
os.system(command6)
os.system(command7)
