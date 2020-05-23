import os

#Test runs on mnist data each two epochs
command1 = "python run.py --config configs/testfiles/test_gaussvae.yaml --experiment_name test_gaussvae"
command2 = "python run.py --config configs/testfiles/test_betavae.yaml --experiment_name test_betavae"
command3 = "python run.py --config configs/testfiles/test_catvae.yaml --experiment_name test_catvae"
command4 = "python run.py --config configs/testfiles/test_infovae.yaml --experiment_name test_infovae"

os.system(command1)
os.system(command2)
os.system(command3)
os.system(command4)

