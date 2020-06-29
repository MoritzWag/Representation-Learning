import os

lambda_digs = [1., 5., 10.]
lambda_offdigs = [1., 5., 10.]
max_epochs = 30

count = 0 
for lambda_dig in lambda_digs:
    for lambda_offdig in lambda_offdigs:
        count += 1
        command = f"python run.py --config configs/ADIDAS/dip_vae.yaml --experiment_name DIPVae --run_name DIPtest_{count} --lambda_dig {lambda_dig} --lambda_offdig {lambda_offdig}"
        
        print(command)
        
        os.system(command)