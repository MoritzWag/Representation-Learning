import os 
import pandas as pd
from library.postprocessing import get_mlflow_results, plot_boxes

command1 = 'python run.py --config configs/ADIDAS/vae.yaml --max_epochs 40 --experiment_name epoch_check --run_name epoch_40'
command2 = 'python run.py --config configs/ADIDAS/vae.yaml --max_epochs 80 --experiment_name epoch_check --run_name epoch_80'
command3 = 'python run.py --config configs/ADIDAS/vae.yaml --max_epochs 120 --experiment_name epoch_check --run_name epoch_120'

os.system(command1)
os.system(command2)
os.system(command3)

mlruns = os.listdir('mlruns')
latest_mlflow_id = 1

get_mlflow_results(mlflow_id=latest_mlflow_id)