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
import pickle as pkl

from sys import getsizeof
from sklearn.preprocessing import LabelEncoder





def parse_args():
    parser = argparse.ArgumentParser(description='adidas')
    parser.add_argument('--path', type=str, default='/home/ubuntu/data/',
                        help='path to store data')
    parser.add_argument('--test_ratio', type=float, default=0.2,
                        metavar='N', help='testratio for train/test split (default: 0.2)')
    args = parser.parse_args()
    return args

def get_data(args):
    #pdb.set_trace()

    url = "https://syncandshare.lrz.de/dl/fiGkTna5QwQtZRnWcPb8Ch9X/.zip"
    #url = "https://syncandshare.lrz.de/download/MlJiRHRvWkFZc3M1MngzdDNTcmE5/Data/Image_Dataset/image_tensors.zip"
    data_path = os.path.expanduser(args.path)
    storage_path = '{}/adidas/'.format(data_path)
    if not os.path.exists(storage_path):
        #os.mkdir(storage_path)
        #pdb.set_trace()
        os.makedirs(storage_path)


        r = requests.get(url)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall(storage_path)
    
        img_path = f'{storage_path}img/'
        if not os.path.exists(img_path):
            os.mkdir(img_path)
    
        with zipfile.ZipFile(f'{storage_path}Data/Image_Dataset/image_tensors.zip', "r") as f:
            f.extractall(img_path)
    
        img_path = f'{storage_path}img/lmu_practicals/'
        img_unzipped = f'{storage_path}img/img/'
        if not os.path.exists(img_unzipped):
            os.mkdir(img_unzipped)

        files_list = os.listdir(img_path)
        for file in tqdm(files_list):
            with gzip.open(img_path + '/' + file, 'rb') as f_in:
                with open(img_unzipped + '/' + file[:-3], 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
    
        array = []
        for file in tqdm(files_list):
            arr = np.load(img_unzipped + '/' + file[:-3], allow_pickle=True)

            array.append(arr)
        
        array = np.stack(array)
        array = np.squeeze(array)

        # some sanity checks
        file_ids_fv = []
        file_ids_bv = []
        file_ids_sl = []
        file_ids_sv = []
        file_ids_tp = []
        fv_count = 0
        bv_count = 0
        sl_count = 0
        sv_count = 0
        tp_count = 0
        for files in files_list:
            if "Front_View" in files:
                fv_count += 1
                file_ids_fv.append(files[0:8])
            elif "Back_View" in files:
                bv_count += 1
                file_ids_bv.append(files[0:8])
            elif "Side_Lateral_View" in files:
                sl_count += 1
                file_ids_sl.append(files[0:8])
            elif "Standard_View" in files:
                sv_count += 1
                file_ids_sv.append(files[0:8])
            elif "Top_Portrait_View" in files:
                tp_count += 1
                file_ids_tp.append(files[0:8])

        duplicate_value = pd.Series(file_ids_sl)[pd.Series(file_ids_sl).duplicated()].values


        # return index in files list
        index_duplicate = [files_list.index(l) for l in files_list if l.startswith(f'{duplicate_value[0]}-Side_Lateral_View')]
        
        array = np.delete(array, index_duplicate[0], axis=0)

        # save array as npy
        idx_train = np.random.choice(a=array.shape[0],
                                    size=int((1-args.test_ratio)*array.shape[0]),
                                    replace=False)
        idx_test = [a for a in range(array.shape[0]) if not a in idx_train]

        X_train = array[idx_train, :, :, :]
        X_test = array[idx_test, :, :, :]
        
        # read in meta data in pd.DF format
        meta_data = pd.read_csv(f"{storage_path}Data/Image_Dataset/adidas_lmu_practicals_image_data.csv", sep=";")
        
        # list of unique ids in numpy.array
        unique_file_ids = set(files_list) 

        # list of unique ids in meta_data
        meta_ids = meta_data.ARTICLE_NUMBER.unique()
        overview_meta_uniques = meta_data["PRODUCT_VIEW"].value_counts()

        # there is mismatch for the side_lateral_views 
        meta_sl = meta_data[meta_data['PRODUCT_VIEW'] == 'Side Lateral View']

        meta_train = meta_data.iloc[idx_train, :]
        meta_test = meta_data.iloc[idx_test, :]

        meta_train.reset_index(drop=True, inplace=True)
        meta_test.reset_index(drop=True, inplace=True)
        # get indices for each view and train/test split
        # train
        front_view_idx_train = meta_train.index[meta_train['PRODUCT_VIEW'] == 'Front View'].tolist()
        back_view_idx_train = meta_train.index[meta_train['PRODUCT_VIEW'] == 'Back View'].tolist()
        side_lateral_idx_train = meta_train.index[meta_train['PRODUCT_VIEW'] == 'Side Lateral View'].tolist()
        standard_view_idx_train = meta_train.index[meta_train['PRODUCT_VIEW'] == 'Standard View'].tolist()
        top_portrait_idx_train = meta_train.index[meta_train['PRODUCT_VIEW'] == 'Top Portrait View'].tolist()
        # test
        front_view_idx_test = meta_test.index[meta_test['PRODUCT_VIEW'] == 'Front View'].tolist()
        back_view_idx_test = meta_test.index[meta_test['PRODUCT_VIEW'] == 'Back View'].tolist()
        side_lateral_idx_test = meta_test.index[meta_test['PRODUCT_VIEW'] == 'Side Lateral View'].tolist()
        standard_view_idx_test = meta_test.index[meta_test['PRODUCT_VIEW'] == 'Standard View'].tolist()
        top_portrait_idx_test = meta_test.index[meta_test['PRODUCT_VIEW'] == 'Top Portrait View'].tolist()

        # get indeces for each view
        #front_view_idx = meta_data.index[meta_data['PRODUCT_VIEW'] == 'Front View'].tolist()
        #back_view_idx = meta_data.index[meta_data['PRODCUT_VIEW'] == 'Back View'].tolist()
        #side_lateral_idx = meta_data.index[meta_data['PRODUCT_VIEW'] == 'Side Lateral View'].tolist()
        #standard_view_idx = meta_data.index[meta_data['PRODUCT_VIEW'] == 'Standard View'].tolist()
        #top_portrait_idx = meta_data.index[meta_data['PRODUCT_VIEW'] == 'Top Portrait View'].tolist()

        X_train_front_view = X_train[front_view_idx_train, :, :, :]
        X_train_back_view = X_train[back_view_idx_train, :, :, :]
        X_train_side_lateral = X_train[side_lateral_idx_train, :, :, :]
        X_train_standard_view = X_train[standard_view_idx_train, :, :, :]
        X_train_top_portrait = X_train[top_portrait_idx_train, :, :, :]

        X_test_front_view = X_test[front_view_idx_test, :, :, :]
        X_test_back_view = X_test[back_view_idx_test, :, :, :]
        X_test_side_lateral = X_test[side_lateral_idx_test, :, :, :]
        X_test_standard_view = X_test[standard_view_idx_test, :, :, :]
        X_test_top_portrait = X_test[top_portrait_idx_test, :, :, :]
        
        os.system(f'rm -rf {img_unzipped}')
        os.system(f'rm -rf {img_path}')


        save_path = f"{storage_path}Data/"

        np.save(file='{}X_train_front_view'.format(save_path), arr=X_train_front_view)
        np.save(file='{}X_train_back_view'.format(save_path), arr=X_train_back_view)
        np.save(file='{}X_train_side_lateral'.format(save_path), arr=X_train_side_lateral)
        np.save(file='{}X_train_standard_view'.format(save_path), arr=X_train_standard_view)
        np.save(file='{}X_train_top_portrait'.format(save_path), arr=X_train_top_portrait)

        
        np.save(file='{}X_test_front_view'.format(save_path), arr=X_test_front_view)
        np.save(file='{}X_test_back_view'.format(save_path), arr=X_test_back_view)
        np.save(file='{}X_test_side_lateral'.format(save_path), arr=X_test_side_lateral)
        np.save(file='{}X_test_standard_view'.format(save_path), arr=X_test_standard_view)
        np.save(file='{}X_test_top_portrait'.format(save_path), arr=X_test_top_portrait)


        ## save data
        save_path = f'{storage_path}Data/'
    
        meta_data_image = pd.read_csv(f"{storage_path}Data/Image_Dataset/adidas_lmu_practicals_image_data.csv", sep=";")
        meta_data_all = pd.read_csv(f"{storage_path}Data/adidas_lmu_practicals_data.csv", sep=",")
        
        meta_data = pd.merge(meta_data_image, meta_data_all, how="left", on=['ARTICLE_NUMBER'])


        labels = meta_data['ASSET_CATEGORY'].values

        Y = pd.DataFrame(labels, columns=['label'])
        le = LabelEncoder()
        le.fit(Y['label'].values)
        le.transform(Y['label'].values)
        Y_foo = le.transform(Y['label'].values)
        
        Y = pd.DataFrame(Y_foo.tolist(), columns=['label'])

        Y_train = Y.iloc[idx_train, :]
        Y_test = Y.iloc[idx_test, :]

        Y_train_front_view = Y_train.iloc[front_view_idx_train, :]
        Y_train_back_view = Y_train.iloc[back_view_idx_train, :]
        Y_train_side_lateral = Y_train.iloc[side_lateral_idx_train, :]
        Y_train_standard_view = Y_train.iloc[standard_view_idx_train, :]
        Y_train_top_portrait = Y_train.iloc[top_portrait_idx_train, :]

        Y_train_front_view.to_csv('{}Y_train_front_view.csv'.format(save_path))
        Y_train_back_view.to_csv('{}Y_train_back_view.csv'.format(save_path))
        Y_train_side_lateral.to_csv('{}Y_train_side_lateral.csv'.format(save_path))
        Y_train_standard_view.to_csv('{}Y_train_standard_view.csv'.format(save_path))
        Y_train_top_portrait.to_csv('{}Y_train_top_portrait.csv'.format(save_path))

        Y_test_front_view = Y_test.iloc[front_view_idx_test, :]
        Y_test_back_view = Y_test.iloc[back_view_idx_test, :]
        Y_test_side_lateral = Y_test.iloc[side_lateral_idx_test, :]
        Y_test_standard_view = Y_test.iloc[standard_view_idx_test, :]
        Y_test_top_portrait = Y_test.iloc[top_portrait_idx_test, :]

        Y_test_front_view.to_csv('{}Y_test_front_view.csv'.format(save_path))
        Y_test_back_view.to_csv('{}Y_test_back_view.csv'.format(save_path))
        Y_test_side_lateral.to_csv('{}Y_test_side_lateral.csv'.format(save_path))
        Y_test_standard_view.to_csv('{}Y_test_standard_view.csv'.format(save_path))
        Y_test_top_portrait.to_csv('{}Y_test_top_portrait.csv'.format(save_path))


if __name__ == "__main__":
    args = parse_args()
    get_data(args=args)