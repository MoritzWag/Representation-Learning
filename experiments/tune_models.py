import os

# tune beta-VAE
os.system('python ../tuner.py --config ../configs/ADIDAS/beta_vae.yaml --run_name beta_vae_hb --experiment_name beta_vae_hb')

# tune DIP-VAE
os.system('python ../tuner.py --config ../configs/ADIDAS/dip_vae.yaml --run_name dip_vae_hb --experiment_name dip_vae_hb')

# tune CatVAE
os.system('python ../tuner.py --config ../configs/ADIDAS/gaussmix_vae.yaml --run_name cat_vae_hb --experiment_name cat_vae_hb')

# tune InfoVAE
os.system('python ../tuner.py --config ../configs/ADIDAS/info_vae.yaml --run_name info_vae_hb --experiment_name info_vae_hb')