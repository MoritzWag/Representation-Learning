import os

#MNIST
command1 = "python run.py --config configs/MNIST/vae.yaml --experiment_name VaeGaussian"
command2 = "python run.py --config configs/MNIST/info_vae.yaml --experiment_name InfoVae"
command3 = "python run.py --config configs/MNIST/cat_vae.yaml --experiment_name CatVae"

#FASHIONMNIST
command4 = "python run.py --config configs/FASHIONMNIST/vae.yaml --experiment_name VaeGaussian"
command5 = "python run.py --config configs/FASHIONMNIST/info_vae.yaml --experiment_name InfoVae"
command6 = "python run.py --config configs/FASHIONMNIST/cat_vae.yaml --experiment_name CatVae"



os.system(command1)
os.system(command2)
os.system(command3)
os.system(command4)
os.system(command5)
os.system(command6)
