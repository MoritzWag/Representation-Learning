import os 
import pandas as pd
from library.postprocessing import get_mlflow_results, plot_boxes

command1 = 'python run.py --config configs/ADIDAS/vae.yaml --max_epochs 80 --latent_dim 5 --experiment_name latent_check --run_name latent_5'
command2 = 'python run.py --config configs/ADIDAS/vae.yaml --max_epochs 80 --latent_dim 10 --experiment_name latent_check --run_name latent_10'
command3 = 'python run.py --config configs/ADIDAS/vae.yaml --max_epochs 80 --latent_dim 20 --experiment_name latent_check --run_name latent_20'
command4 = 'python run.py --config configs/ADIDAS/vae.yaml --max_epochs 80 --latent_dim 40 --experiment_name latent_check --run_name latent_40'

os.system(command1)
os.system(command2)
os.system(command3)
os.system(command4)

mlruns = os.listdir('mlruns')
latest_mlflow_id = int(max(mlruns))

get_mlflow_results(mlflow_id=latest_mlflow_id)



