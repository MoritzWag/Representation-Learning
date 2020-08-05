import os 
import pandas as pd
from library.postprocessing import get_mlflow_results, plot_boxes

command1 = 'python run.py --config configs/ADIDAS/vae.yaml --max_epochs 40 --kld_weight 0.0008 --experiment_name kld_check --run_name kld_weight_00001'
command2 = 'python run.py --config configs/ADIDAS/vae.yaml --max_epochs 40 --kld_weight 0.01 --experiment_name kld_check --run_name kld_weight_001'
command3 = 'python run.py --config configs/ADIDAS/vae.yaml --max_epochs 40 --kld_weight 0.1 --experiment_name kld_check --run_name kld_weight_01'
command4 = 'python run.py --config configs/ADIDAS/vae.yaml --max_epochs 40 --kld_weight 0.5 --experiment_name kld_check --run_name kld_weight_05'
command5 = 'python run.py --config configs/ADIDAS/vae.yaml --max_epochs 40 --kld_weight 0.75 --experiment_name kld_check --run_name kld_weight_075'
command6 = 'python run.py --config configs/ADIDAS/vae.yaml --max_epochs 40 --kld_weight 1. --experiment_name kld_check --run_name kld_weight_1'

os.system(command1)
os.system(command2)
os.system(command3)
os.system(command4)
os.system(command5)
os.system(command6)

mlruns = os.listdir('mlruns')
latest_mlflow_id = 1

get_mlflow_results(mlflow_id=latest_mlflow_id)