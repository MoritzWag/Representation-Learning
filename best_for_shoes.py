import os 
from library.postprocessing import get_mlflow_results

experiment_name ="shoes"
max_epochs = 80


command_vae = f"python run.py --config configs/ADIDAS/vae.yaml --experiment_name {experiment_name} --run_name vanilla_vae --max_epochs {max_epochs}"
command_beta = f"python run.py --config configs/ADIDAS/beta_vae.yaml --experiment_name {experiment_name} --run_name beta_vae --max_epochs {max_epochs}"
command_cat = f"python run.py --config configs/ADIDAS/gaussmix_vae.yaml --experiment_name {experiment_name} --run_name cat_vae --max_epochs {max_epochs}"


os.system(command_vae)
os.system(command_beta)
os.system(command_cat)



get_mlflow_results(mlflow_id=33)