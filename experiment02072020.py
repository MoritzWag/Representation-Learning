import os

experiment_name = 'rf_exp_comp'
num_epochs = 40

command1 = f"python run.py --config configs/ADIDAS/pca.yaml --latent_dim 10 --experiment_name {experiment_name} --run_name pca --max_epochs 4"
command2 = f"python run.py --config configs/ADIDAS/beta_vae.yaml --latent_dim 10 --experiment_name {experiment_name} --run_name vae --max_epochs {num_epochs}"
command3 = f"python run.py --config configs/ADIDAS/beta_vae.yaml --latent_dim 10 --experiment_name {experiment_name} --run_name beta --max_epochs {num_epochs}"
command4 = f"python run.py --config configs/ADIDAS/dip_vae.yaml --latent_dim 10 --experiment_name {experiment_name} --run_name beta --max_epochs {num_epochs}"

os.system(command1)
os.system(command2)
os.system(command3)
os.system(command4)
