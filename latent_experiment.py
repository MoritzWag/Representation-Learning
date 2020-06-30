import os

latent_dims = [5, 10, 15]

for latent_dim in latent_dims:
    command = f"python run.py --config configs/ADIDAS/vae.yaml --experiment_name  latents --run_name latent_{latent_dim} --latent_dim {latent_dim} --max_epochs 50"
    
    print(command)
    
    os.system(command)