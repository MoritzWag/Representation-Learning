import subprocess 
import zipfile
import tarfile
import requests
from tqdm import tqdm

import numpy as np 
import pandas as pd 
import argparse

import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch
import torchvision
import os 
import pdb

def parse_args():
    parser = argparse.ArgumentParser(description='celeba')
    parser.add_argument('--path', type=str, default = '.',

                        help='path to store data')
    args = parser.parse_args()
    return args

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()

    response = session.get(URL, params={ 'id': id }, stream=True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def save_response_content(response, destination, chunk_size=32*1024):
    total_size = int(response.headers.get('content-length', 0))
    with open(destination, "wb") as f:
        for chunk in tqdm(response.iter_content(chunk_size), total=total_size,
                        unit='B', unit_scale=True, desc=destination):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

def unzip(filepath):
    print("Extracting: " + filepath)
    dirpath = os.path.dirname(filepath)
    with zipfile.ZipFile(filepath) as zf:
        zf.extractall(dirpath)
    os.remove(filepath)



def get_data(args):
    pdb.set_trace()
    data_dir = 'celeba'
    data_path = os.path.expanduser(args.path)
    storage_path = '{}/celeba/'.format(data_path)
    if not os.path.exists(storage_path):
        os.makedirs(storage_path)
    
    filename, drive_id  = "img_align_celeba_png.7z", "https://drive.google.com/drive/folders/0B7EVK8r0v71peklHb0pGdDl6R28"
    save_path = f"{storage_path}{filename}"
    #save_path = os.path.join(data_dir, filename)
    if os.path.exists(save_path):
        print('[*] {} already exists'.format(save_path))
    else:
        download_file_from_google_drive(drive_id, save_path)
    
    zip_dir = ''
    with zipfile.ZipFile(save_path) as zf:
        zip_dir = zf.namelist()[0]
        zf.extractall(data_dir)
    os.remove(save_path)
    #os.rename(os.path.join(data_dir, zip_dir), os.path.join(dirpath, data_dir))


if __name__ == "__main__":
    args = parse_args()
    get_data(args=args)