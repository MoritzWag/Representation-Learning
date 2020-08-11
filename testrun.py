import os

#Test runs
command1 = "python run.py --config configs/ADIDAS/vae.yaml --experiment_name test --run_name vae --max_epochs 1"
command2 = "python run.py --config configs/ADIDAS/beta_vae.yaml --experiment_name test --run_name vae_beta --max_epochs 1"
command3 = "python run.py --config configs/ADIDAS/dip_vae.yaml --experiment_name test --run_name vae_dip --max_epochs 1"
command4 = "python run.py --config configs/ADIDAS/info_vae.yaml --experiment_name test --run_name vae_info --max_epochs 1"
command5 = "python run.py --config configs/ADIDAS/gaussmix_vae.yaml --experiment_name test --run_name vae_gaussmix --max_epochs 1"

os.system(command1)
os.system(command2)
os.system(command3)
os.system(command4)
os.system(command5)

