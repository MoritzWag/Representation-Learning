import os

#cifar10
command1 = "python run.py --config configs/cifar10/vae_cifar10.yaml --experiment_name VaeGaussian"
command2 = "python run.py --config configs/cifar10/vae_info.yaml --experiment_name InfoVae"
command3 = "python run.py --config configs/cifar10/cat_vae.yaml --experiment_name CatVae"

#adidas
command4 = "python run.py --config configs/adidas/vae_adidas.yaml --experiment_name InfoVae"
#command5 = "python run.py --config configs/adidas/info_vae.yaml --experiment_name InfoVae"
#command6 = "python run.py --config configs/adidas/cat_vae.yaml --experiment_name CatVae"



#os.system(command1)
os.system(command2)
os.system(command3)
#os.system(command4)
os.system(command5)
os.system(command6)