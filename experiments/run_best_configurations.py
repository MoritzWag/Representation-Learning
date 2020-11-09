import os 
from library.postprocessing import get_mlflow_results


experiment_name = "best_configurations"
max_epochs = 80

command_vae = f"python ../run.py --config configs/ADIDAS/vae.yaml --experiment_name {experiment_name} --run_name vanilla_vae --max_epochs {max_epochs}"
command_beta = f"python ../run.py --config configs/ADIDAS/beta_vae.yaml --experiment_name {experiment_name} --run_name beta_vae --max_epochs {max_epochs}"
command_info = f"python ../run.py --config configs/ADIDAS/info_vae.yaml --experiment_name {experiment_name} --run_name info_vae --max_epochs {max_epochs}"
command_dip = f"python ../run.py --config configs/ADIDAS/dip_vae.yaml --experiment_name {experiment_name} --run_name dip_vae --max_epochs {max_epochs}"
command_cat = f"python ../run.py --config configs/ADIDAS/gaussmix_vae.yaml --experiment_name {experiment_name} --run_name cat_vae --max_epochs {max_epochs}"
command_pca = f"python ../run.py --config configs/ADIDAS/pca.yaml --experiment_name {experiment_name} --run_name pca --max_epochs 4"
command_autoencoder = f"python ../run.py --config configs/ADIDAS/autoenc.yaml --experiment_name {experiment_name} --run_name autoencoder --max_epochs {max_epochs}"



os.system(command_vae)
os.system(command_beta)
os.system(command_info)
os.system(command_dip)
os.system(command_cat) 
os.system(command_pca) 
os.system(command_autoencoder)



mlruns = os.listdir('mlruns')
latest_mlflow_id = int(max(mlruns))

get_mlflow_results(mlflow_id=latest_mlflow_id)