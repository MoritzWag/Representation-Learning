
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
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder


def parse_args():
    parser = argparse.ArgumentParser(description='adidas')
    parser.add_argument('--path', type=str, default='/home/ubuntu/data',
                        help='path to store data')
    parser.add_argument('--test_ratio', type=float, default=0.2,
                        metavar='N', help='testratio for train/test split (default: 0.2)')
    args = parser.parse_args()
    return args

def get_data(args):
    
    url = "https://syncandshare.lrz.de/dl/fiGkTna5QwQtZRnWcPb8Ch9X/.zip"
    #url = "https://syncandshare.lrz.de/download/MlJiRHRvWkFZc3M1MngzdDNTcmE5/Data/Image_Dataset/image_tensors.zip"
    data_path = os.path.expanduser(args.path)
    storage_path = '{}/adidas/'.format(data_path)
    #if not os.path.exists(storage_path):
    if  not os.path.exists(storage_path):
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

        os.system(f'rm -rf {img_path}')

        ## prepare meta data


        meta_data_all = pd.read_csv(f"{storage_path}attributes.csv", sep=",")

        meta_data_all = meta_data_all.drop(columns=['ASSET_CATEGORY', 'PRICE', 'COLOR_GRP_2', 'COLOR_GRP_3',
                                    'COLOR_GRP_4', 'BUSINESS_UNIT_DESCR', 'PRODUCT_FRANCHISE_DESCR',
                                    'SPORTS_CATEGORY_DESCR', 'SALES_LINE_DESCR'])


        

        # remove all variables with too many NaNs
        #meta_data_all = meta_data_all.iloc[:, :-4]
        #meta_data_all = meta_data_all.drop(columns=['COLOR_GRP_2_x', 'COLOR_GRP_3_x', 'COLOR_GRP_4_x', 'GENDER_y', 
        #                                            'AGE_GROUP_y', 'PRICE_y', 'PRODUCT_DIVISION_DESCR_y', 'COLOR_GRP_1_y',
        #                                            'BUSINESS_UNIT_DESCR', 'SEASON', 'WEEK', 'SALES_LINE_DESCR',
        #                                            'CAMPAIGN', 'TOTAL_MARKDOWN_PCT', 'PRODUCT_FRANCHISE_DESCR',
        #                                            'COLOR_GRP_2_y', 'COLOR_GRP_3_y', 'COLOR_GRP_4_y', 'SOLD_QTY', 'PRICE_x',
        #                                            'indicator_column'])

        # determine remaining missing values
        data_missings = meta_data_all[meta_data_all.isnull().any(axis=1)]
        tids_missing = data_missings.ARTICLE_NUMBER.tolist()
        #meta_data_all = meta_data_all.dropna(how='any', subset=['COLOR_GRP_1_x'])

        # def prepare_attributes(data, cols):
        #     remaining_data = data.drop(cols, axis=1)

        #     categorical_data = data[cols]
        #     oe = OrdinalEncoder()
        #     oe.fit(categorical_data)
        #     categorical_data = oe.transform(categorical_data)

        #     categorical_data = pd.DataFrame(categorical_data, columns=cols)

        #     df_all = pd.concat([remaining_data, categorical_data], axis=1, ignore_index=False)

        #     return df_all

        def prepare_attributes(data, cols):
            remaining_data = data.drop(cols, axis=1)
            categorical_data = data[cols]
            transformed_data = categorical_data.apply(LabelEncoder().fit_transform)

            df_all = pd.concat([remaining_data, transformed_data], axis=1, ignore_index=False)

            return df_all
        
        meta_data_all = meta_data_all.dropna(how='any', subset=['COLOR_GRP_1'])

        categorical_features = ['PRODUCT_GROUP_DESCR','PRODUCT_DIVISION_DESCR', 'GENDER', 
                                'AGE_GROUP', 'COLOR_GRP_1', 'PRODUCT_TYPE_DESCR']
        
        #categorical_features = ['ASSET_CATEGORY', 'PRODUCT_DIVISION_DESCR_x',
        #                        'GENDER_x', 'AGE_GROUP_x', 'PRICE_x',
        #                        'COLOR_GRP_1_x', 'SEASON', 'PRODUCT_GROUP_DESCR',
        #                        'PRODUCT_TYPE_DESCR', 'PRODUCT_FRANCHISE_DESCR',
        #                        'SPORTS_CATEGORY_DESCR']
        meta_data = prepare_attributes(data=meta_data_all, cols=categorical_features)    

        #meta_data = meta_data.dropna(how='any', subset=['COLOR_GRP_1'])    
        #meta_data = meta_data.dropna(how='any', subset=['COLOR_GRP_1_x'])

        array_fv = []
        array_bv = []
        array_sl = []
        array_sv = []
        array_tp = []

        file_ids_fv = []
        file_ids_bv = []
        file_ids_sl = []
        file_ids_sv = []
        file_ids_tp = []


        for file in tqdm(files_list):
            arr = np.load(img_unzipped + '/' + file[:-3], allow_pickle=True)

            if file[0:8] in tids_missing:
                print("too many missings for this image")
                continue
        
            if "Front_View" in file:
                array_fv.append(arr)
                file_ids_fv.append(file[0:8])
            if "Back_View" in file:
                array_bv.append(arr)
                file_ids_bv.append(file[0:8])
            if "Side_Lateral_View" in file:
                array_sl.append(arr)
                file_ids_sl.append(file[0:8])
            if "Standard_View" in file:
                array_sv.append(arr)
                file_ids_sv.append(file[0:8])
            if "Top_Portrait_View" in file:
                array_tp.append(arr)
                file_ids_tp.append(file[0:8])

        os.system(f'rm -rf {img_unzipped}')
        
        array_fv = np.stack(array_fv)
        array_fv = np.squeeze(array_fv)

        array_bv = np.stack(array_bv)
        array_bv = np.squeeze(array_bv)

        array_sl = np.stack(array_sl)
        array_sl = np.squeeze(array_sl)

        array_sv = np.stack(array_sv)
        array_sv = np.squeeze(array_sv)

        array_tp = np.stack(array_tp)
        array_tp = np.squeeze(array_tp)

        fv_list = [array_fv, file_ids_fv]
        bv_list = [array_bv, file_ids_bv]
        sl_list = [array_sl, file_ids_sl]
        sv_list = [array_sv, file_ids_sv]
        tp_list = [array_tp, file_ids_tp]

        meta_data = meta_data.reset_index()
        
        #meta_data = meta_data.rename(columns={'PRODUCT_GROUP_DESCR': 'labels'})

        # get indeces for each view
        front_view_idx = meta_data.index[meta_data['PRODUCT_VIEW'] == 'Front View'].tolist()
        back_view_idx = meta_data.index[meta_data['PRODUCT_VIEW'] == 'Back View'].tolist()
        side_lateral_idx = meta_data.index[meta_data['PRODUCT_VIEW'] == 'Side Lateral View'].tolist()
        standard_view_idx = meta_data.index[meta_data['PRODUCT_VIEW'] == 'Standard View'].tolist()
        top_portrait_idx = meta_data.index[meta_data['PRODUCT_VIEW'] == 'Top Portrait View'].tolist()

        #features = ['SOLD_QTY_SUM', 'SOLD_QTY_AVG']
        #features.extend(categorical_features)


        meta_data_fv = meta_data.iloc[front_view_idx, :]
        meta_data_bv = meta_data.iloc[back_view_idx, :]
        meta_data_sl = meta_data.iloc[side_lateral_idx, :]
        meta_data_sv = meta_data.iloc[standard_view_idx, :]
        meta_data_tp = meta_data.iloc[top_portrait_idx, :]

        # sort meta data so that it matches with the image arrays.
        meta_data_fv = meta_data_fv.set_index('ARTICLE_NUMBER')
        meta_data_bv = meta_data_bv.set_index('ARTICLE_NUMBER')
        meta_data_sl = meta_data_sl.set_index('ARTICLE_NUMBER')
        meta_data_sv = meta_data_sv.set_index('ARTICLE_NUMBER')
        meta_data_tp = meta_data_tp.set_index('ARTICLE_NUMBER')


        meta_data_fv = meta_data_fv.loc[file_ids_fv]
        meta_data_bv = meta_data_bv.loc[file_ids_bv]
        meta_data_sl = meta_data_sl.loc[file_ids_sl]
        meta_data_sv = meta_data_sv.loc[file_ids_sv]
        meta_data_tp = meta_data_tp.loc[file_ids_tp]


        meta_data_fv.reset_index(drop=True, inplace=True)
        meta_data_bv.reset_index(drop=True, inplace=True)
        meta_data_sl.reset_index(drop=True, inplace=True)
        meta_data_sv.reset_index(drop=True, inplace=True)
        meta_data_tp.reset_index(drop=True, inplace=True)

        meta_data_fv = meta_data_fv[categorical_features]
        #meta_data_fv = meta_data_fv.rename(columns={'PRODUCT_GROUP_DESCR': 'labels'})

        meta_data_bv = meta_data_bv[categorical_features]
        #meta_data_bv = meta_data_bv.rename(columns={'PRODUCT_GROUP_DESCR': 'labels'})

        meta_data_sl = meta_data_sl[categorical_features]
        #meta_data_sl = meta_data_sl.rename(columns={'PRODUCT_GROUP_DESCR': 'labels'})

        meta_data_sv = meta_data_sv[categorical_features]
        #meta_data_sv = meta_data_sv.rename(columns={'PRODUCT_GROUP_DESCR': 'labels'})

        meta_data_tp = meta_data_tp[categorical_features]
        #meta_data_tp = meta_data_tp.rename(columns={'PRODUCT_GROUP_DESCR': 'labels'})

        

        # train / test split
        idx_train_fv = np.random.choice(a=array_fv.shape[0],
                                    size=int((1-args.test_ratio)*array_fv.shape[0]),
                                    replace=False)
        idx_test_fv = [a for a in range(array_fv.shape[0]) if not a in idx_train_fv]
        
        idx_train_bv = np.random.choice(a=array_bv.shape[0],
                                    size=int((1-args.test_ratio)*array_bv.shape[0]),
                                    replace=False)
        idx_test_bv = [a for a in range(array_bv.shape[0]) if not a in idx_train_bv]
        
        idx_train_sl = np.random.choice(a=array_sl.shape[0],
                                    size=int((1-args.test_ratio)*array_sl.shape[0]),
                                    replace=False)
        idx_test_sl = [a for a in range(array_sl.shape[0]) if not a in idx_train_sl]

        idx_train_sv = np.random.choice(a=array_sv.shape[0],
                                    size=int((1-args.test_ratio)*array_sv.shape[0]),
                                    replace=False)
        idx_test_sv = [a for a in range(array_sv.shape[0]) if not a in idx_train_sv]

        idx_train_tp = np.random.choice(a=array_tp.shape[0],
                                    size=int((1-args.test_ratio)*array_tp.shape[0]),
                                    replace=False)
        idx_test_tp = [a for a in range(array_tp.shape[0]) if not a in idx_train_tp]
        

        
        X_train_front_view = array_fv[idx_train_fv, :, :, :]
        X_test_front_view = array_fv[idx_test_fv, :, :, :]

        X_train_back_view = array_bv[idx_train_bv, :, :, :]
        X_test_back_view = array_bv[idx_test_bv, :, :, :]

        X_train_side_lateral = array_sl[idx_train_sl, :, :, :]
        X_test_side_lateral = array_sl[idx_test_sl, :, :, :]

        X_train_standard_view = array_sv[idx_train_sv, :, :, :]
        X_test_standard_view = array_sv[idx_test_sv, :, :, :]

        X_train_top_portrait = array_tp[idx_train_tp, :, :, :]
        X_test_top_portrait = array_tp[idx_test_tp, :, :, :]

        Y_train_front_view = meta_data_fv.iloc[idx_train_fv, :]
        Y_test_front_view = meta_data_fv.iloc[idx_test_fv, :]

        Y_train_back_view = meta_data_bv.iloc[idx_train_bv, :]
        Y_test_back_view = meta_data_bv.iloc[idx_test_bv, :]

        Y_train_side_lateral = meta_data_sl.iloc[idx_train_sl, :]
        Y_test_side_lateral = meta_data_sl.iloc[idx_test_sl, :]

        Y_train_standard_view = meta_data_sv.iloc[idx_train_sv, :]
        Y_test_standard_view = meta_data_sv.iloc[idx_test_sv, :]

        Y_train_top_portrait = meta_data_tp.iloc[idx_train_tp, :]
        Y_test_top_portrait = meta_data_tp.iloc[idx_test_tp, :]

        # save X and Y; training and test data - for each view seperately
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

        Y_train_front_view.to_csv('{}Y_train_front_view.csv'.format(save_path))
        Y_train_back_view.to_csv('{}Y_train_back_view.csv'.format(save_path))
        Y_train_side_lateral.to_csv('{}Y_train_side_lateral.csv'.format(save_path))
        Y_train_standard_view.to_csv('{}Y_train_standard_view.csv'.format(save_path))
        Y_train_top_portrait.to_csv('{}Y_train_top_portrait.csv'.format(save_path))

        Y_test_front_view.to_csv('{}Y_test_front_view.csv'.format(save_path))
        Y_test_back_view.to_csv('{}Y_test_back_view.csv'.format(save_path))
        Y_test_side_lateral.to_csv('{}Y_test_side_lateral.csv'.format(save_path))
        Y_test_standard_view.to_csv('{}Y_test_standard_view.csv'.format(save_path))
        Y_test_top_portrait.to_csv('{}Y_test_top_portrait.csv'.format(save_path))



if __name__ == "__main__":
    args = parse_args()
    get_data(args=args)