import os 
from library.postprocessing import get_mlflow_results

experiment_name = "check_cat"
max_epochs = 80 

command_cat_1 = f"python run.py --config configs/ADIDAS/gaussmix_vae.yaml --experiment_name {experiment_name} --run_name cat_vae_1st --max_epochs {max_epochs} --anneal_rate 0.5 --cat_weight 2 --cont_weight 1 --temperature 1"
command_cat_2 = f"python run.py --config configs/ADIDAS/gaussmix_vae.yaml --experiment_name {experiment_name} --run_name cat_vae_2nd --max_epochs {max_epochs} --anneal_rate 0.3 --cat_weight 2 --cont_weight 1 --temperature 11"
command_cat_3 = f"python run.py --config configs/ADIDAS/gaussmix_vae.yaml --experiment_name {experiment_name} --run_name cat_vae_3rd --max_epochs {max_epochs} --anneal_rate 0.4 --cat_weight 1 --cont_weight 1 --temperature 31"


os.system(command_cat_1)
os.system(command_cat_2)
os.system(command_cat_3)

get_mlflow_results(mlflow_id=29)