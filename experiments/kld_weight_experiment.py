import os 
import pandas as pd
from library.postprocessing import get_mlflow_results

command1 = 'python ../run.py --config ../configs/ADIDAS/vae.yaml --max_epochs 40 --kld_weight 0.00001 --experiment_name kld_check --run_name kld_weight_00001'
command2 = 'python ../run.py --config ../configs/ADIDAS/vae.yaml --max_epochs 40 --kld_weight 0.0001 --experiment_name kld_check --run_name kld_weight_0001'
command3 = 'python ../run.py --config ../configs/ADIDAS/vae.yaml --max_epochs 40 --kld_weight 0.0005 --experiment_name kld_check --run_name kld_weight_0005'
command4 = 'python ../run.py --config ../configs/ADIDAS/vae.yaml --max_epochs 40 --kld_weight 0.001 --experiment_name kld_check --run_name kld_weight_001'
command5 = 'python ../run.py --config ../configs/ADIDAS/vae.yaml --max_epochs 40 --kld_weight 0.01 --experiment_name kld_check --run_name kld_weight_01'
command6 = 'python ../run.py --config ../configs/ADIDAS/vae.yaml --max_epochs 40 --kld_weight 0.1 --experiment_name kld_check --run_name kld_weight_1'
command7 = 'python ../run.py --config ../configs/ADIDAS/vae.yaml --max_epochs 40 --kld_weight 0.5 --experiment_name kld_check --run_name kld_weight_5'
command8 = 'python ../run.py --config ../configs/ADIDAS/vae.yaml --max_epochs 40 --kld_weight 1. --experiment_name kld_check --run_name kld_weight_1_0'

os.system(command1)
os.system(command2)
os.system(command3)
os.system(command4)
os.system(command5)
os.system(command6)
os.system(command7)
os.system(command8)

mlruns = os.listdir('mlruns')
latest_mlflow_id = int(max(mlruns))

get_mlflow_results(mlflow_id=latest_mlflow_id)

