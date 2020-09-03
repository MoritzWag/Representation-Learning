import os

command1 = "python run.py --config configs/ADIDAS/dip_vae.yaml --experiment_name test_dip_kld_weight --run_name kld_weight_1 --max_epochs 80 --kld_weight 1"
command2 = "python run.py --config configs/ADIDAS/dip_vae.yaml --experiment_name test_dip_kld_weight --run_name kld_weight_low --max_epochs 80 --kld_weight 0.0005"


os.system(command1)
os.system(command2)