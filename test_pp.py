import os 
from library.postprocessing import get_mlflow_results

os.system('python run.py --config configs/ADIDAS/beta_vae.yaml --experiment_name testtest --run_name test_beta')
os.system('python run.py --config configs/ADIDAS/dip_vae.yaml --experiment_name testtest --run_name test_dip')

get_mlflow_results(mlflow_id=1)