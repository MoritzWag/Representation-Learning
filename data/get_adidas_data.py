import numpy as np 
import pandas as pd 
import argparse
import zipfile
import pdb
import os
import webbrowser
import requests, io
import gzip
import shutil
from tqdm import tqdm





def parse_args():
    parser = argparse.ArgumentParser(description='adidas')
    parser.add_argument('--path', type=str, default='.',
                        help='path to store data')
    parser.add_argument('--test_ratio', type=float, default=0.2,
                        metavar='N', help='testratio for train/test split (default: 0.2)')
    args = parser.parse_args()
    return args

def get_data(args):

    url = "https://syncandshare.lrz.de/dl/fiGkTna5QwQtZRnWcPb8Ch9X/.zip"
    data_path = os.path.expanduser(args.path)
    storage_path = '{}/adidas/'.format(data_path)
    if not os.path.exists(storage_path):
        os.mkdir(storage_path)

        r = requests.get(url)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall(storage_path)
    
        img_path = 'adidas/img/'
        if not os.path.exists(img_path):
            os.mkdir(img_path)
    
        with zipfile.ZipFile('adidas/Data/Image_Dataset/image_tensors.zip', "r") as f:
            f.extractall(img_path)
    
        img_path = 'adidas/img/image_tensors/'
        img_unzipped = 'adidas/img/img/'
        if not os.path.exists(img_unzipped):
            os.mkdir(img_unzipped)

        files_list = os.listdir(img_path)
        for file in files_list:
            with gzip.open(img_path + '/' + file, 'rb') as f_in:
                with open(img_unzipped + '/' + file[:-3], 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
    
        array = []
        for file in tqdm(files_list):
            arr = np.load(img_unzipped + '/' + file[:-3], allow_pickle=True)
            array.append(arr)
        
        array = np.stack(array)
        array = np.squeeze(array)

        # save array as npy
        idx_train = np.random.choice(a=len(files_list),
                                    size=int((1-args.test_ratio)*len(files_list)),
                                    replace=False)
        idx_test = [a for a in range(len(files_list)) if not a in idx_train]

        X_train = array[idx_train, :, :, :]
        X_test = array[idx_test, :, :, :]

        save_path = 'adidas/Data/'
        np.save(file='{}X_train.npy'.format(save_path), arr=X_train)
        np.save(file='{}X_test.npy'.format(save_path), arr=X_test)

        os.rmdir(path='adidas/img')
    
        # read in meta data in pd.DF format
        meta_data = pd.read_csv(f"{storage_path}/Data/Image_Dataset/adidas_lmu_practicals_image_data.csv", sep=";")

    
        # get indeces for each view
        front_view_idx = meta_data.index[meta_data['PRODUCT_VIEW'] == 'Front View'].tolist()
        back_view_idx = meta_data.index[meta_data['PRODCUT_VIEW'] == 'Back View'].tolist()
        side_lateral_idx = meta_data.index[meta_data['PRODUCT_VIEW'] == 'Side Lateral View'].tolist()
        standard_view_idx = meta_data.index[meta_data['PRODUCT_VIEW'] == 'Standard View'].tolist()
        top_portrait_idx = meta_data.index[meta_data['PRODUCT_VIEW'] == 'Top Portrait View'].tolist()


if __name__ == "__main__":
    args = parse_args()
    get_data(args=args)