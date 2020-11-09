import os
from library.postprocessing import get_mlflow_results


command1 = 'python ../run.py --config ../configs/ADIDAS/vae.yaml --max_epochs 40 --experiment_name epoch_val --run_name epochs40'
command2 = 'python ../run.py --config ../configs/ADIDAS/vae.yaml --max_epochs 80 --experiment_name epoch_val --run_name epochs80'
command3 = 'python ../run.py --config ../configs/ADIDAS/vae.yaml --max_epochs 120 --experiment_name epoch_val --run_name epochs120'

os.system(command1)
os.system(command2)
os.system(command3)

mlruns = os.listdir('mlruns')
latest_mlflow_id = int(max(mlruns))

get_mlflow_results(mlflow_id=latest_mlflow_id)