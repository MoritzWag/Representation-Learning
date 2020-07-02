import os

experiment_name = 'basic_runner2'
num_epochs = 80

command1 = f"python run.py --config configs/ADIDAS/pca.yaml --latent_dim 10 --experiment_name {experiment_name} --run_name pca --max_epochs 10"
command2 = f"python run.py --config configs/ADIDAS/info_vae.yaml --latent_dim 10 --experiment_name {experiment_name} --run_name InfoVae --max_epochs {num_epochs}"

os.system(command1)
os.system(command2)