import os

#Test runs on mnist data each two epochs
command1 = "python run.py --config configs/testfiles/autoenc_exp.yaml --experiment_name autoencoder"
command2 = "python run.py --config configs/testfiles/vae_exp_l1.yaml --experiment_name vae_l1"
command3 = "python run.py --config configs/testfiles/vae_exp_l2.yaml --experiment_name vae_l2"
command4 = "python run.py --config configs/testfiles/vaebeta_exp.yaml --experiment_name vae_beta"

os.system(command1)
os.system(command2)
os.system(command3)
os.system(command4)

