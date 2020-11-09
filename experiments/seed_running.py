import os 
import pandas as pd
from library.postprocessing import get_mlflow_results, plot_boxes


command1 = 'python ../run.py --config ../configs/ADIDAS/vae.yaml --max_epochs 40 --manual_seed 1337 --experiment_name seed_check --run_name seed1'
command2 = 'python ../run.py --config ../configs/ADIDAS/vae.yaml --max_epochs 40 --manual_seed 1337 --experiment_name seed_check --run_name seed1_1'
command3 = 'python ../run.py --config ../configs/ADIDAS/vae.yaml --max_epochs 40 --manual_seed 1337 --experiment_name seed_check --run_name seed1_2'
command4 = 'python ../run.py --config ../configs/ADIDAS/vae.yaml --max_epochs 40 --manual_seed 1337 --experiment_name seed_check --run_name seed1_3'
command5 = 'python ../run.py --config ../configs/ADIDAS/vae.yaml --max_epochs 40 --manual_seed 1337 --experiment_name seed_check --run_name seed1_4'
command6 = 'python ../run.py --config ../configs/ADIDAS/vae.yaml --max_epochs 40 --manual_seed 1337 --experiment_name seed_check --run_name seed1_5'

command7 = 'python ../run.py --config ../configs/ADIDAS/vae.yaml --max_epochs 40 --manual_seed 1789 --experiment_name seed_check --run_name seed2'
command8 = 'python ../run.py --config ../configs/ADIDAS/vae.yaml --max_epochs 40 --manual_seed 1789 --experiment_name seed_check --run_name seed2_1'
command9 = 'python ../run.py --config ../configs/ADIDAS/vae.yaml --max_epochs 40 --manual_seed 1789 --experiment_name seed_check --run_name seed2_2'
command10 = 'python ../run.py --config ../configs/ADIDAS/vae.yaml --max_epochs 40 --manual_seed 1789 --experiment_name seed_check --run_name seed2_3'
command11 = 'python ../run.py --config ../configs/ADIDAS/vae.yaml --max_epochs 40 --manual_seed 1789 --experiment_name seed_check --run_name seed2_4'
command12 = 'python ../run.py --config ../configs/ADIDAS/vae.yaml --max_epochs 40 --manual_seed 1789 --experiment_name seed_check --run_name seed2_5'

command13 = 'python ../run.py --config ../configs/ADIDAS/vae.yaml --max_epochs 40 --manual_seed 5687 --experiment_name seed_check --run_name seed3'
command14 = 'python ../run.py --config ../configs/ADIDAS/vae.yaml --max_epochs 40 --manual_seed 5687 --experiment_name seed_check --run_name seed3_1'
command15 = 'python ../run.py --config ../configs/ADIDAS/vae.yaml --max_epochs 40 --manual_seed 5687 --experiment_name seed_check --run_name seed3_2'
command16 = 'python ../run.py --config ../configs/ADIDAS/vae.yaml --max_epochs 40 --manual_seed 5687 --experiment_name seed_check --run_name seed3_3'
command17 = 'python ../run.py --config ../configs/ADIDAS/vae.yaml --max_epochs 40 --manual_seed 5687 --experiment_name seed_check --run_name seed3_4'
command18 = 'python ../run.py --config ../configs/ADIDAS/vae.yaml --max_epochs 40 --manual_seed 5687 --experiment_name seed_check --run_name seed3_5'


os.system(command1)
os.system(command2)
os.system(command3)
os.system(command4)
os.system(command5)
os.system(command6)
os.system(command7)
os.system(command8)
os.system(command9)
os.system(command10)
os.system(command11)
os.system(command12)
os.system(command13)
os.system(command14)
os.system(command15)
os.system(command16)
os.system(command17)
os.system(command18)


mlruns = os.listdir('mlruns')
latest_mlflow_id = int(max(mlruns))

get_mlflow_results(mlflow_id=latest_mlflow_id)


df = pd.read_csv("seed_check_runs.csv")

plot_boxes(df=df, path=None)
