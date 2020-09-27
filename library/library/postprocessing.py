import pandas as pd 
import numpy as np 
import os 
import pdb 
import matplotlib.pyplot as plt


def get_mlflow_results(mlflow_id, path=None):

    if path is None:
        path = f"mlruns/{mlflow_id}"
    else:
        path = f"{path}/mlruns/{mlflow_id}"
    
    runs = [run for run in os.listdir(path) if len(run) == 32 and not run.startswith('performance')]
    frame = pd.DataFrame(columns=['run_name',
                                'experiment_name',
                                'manual_seed',
                                'num_active_units',
                                'gaussian_total_correlation',
                                'gaussian_wasserstein_correlation',
                                'gaussian_wasserstein_correlation_norm',
                                'avg_test_loss',
                                'rf_acc_color_group',
                                'rf_auc_color_group',
                                'rf_aupr_color_group',
                                'rf_avg_pr_color_group',
                                'knn_acc_color_group',
                                'rf_acc_product_group',
                                'rf_auc_product_group',
                                'rf_aupr_product_group',
                                'rf_avg_pr_product_group',
                                'knn_acc_product_group',
                                'rf_acc_product_type',
                                'rf_auc_product_type',
                                'rf_aupr_product_type',
                                'rf_avg_pr_product_type',
                                'knn_acc_product_type',
                                'rf_acc_gender',
                                'rf_auc_gender',
                                'rf_aupr_gender',
                                'rf_avg_pr_gender',
                                'knn_acc_gender',
                                'rf_acc_age_group',
                                'rf_auc_age_group',
                                'rf_aupr_age_group',
                                'rf_avg_pr_age_group',
                                'knn_acc_age_group',
                                'dst_rf_acc',
                                'dst_rf_auc',
                                'dst_rf_aupr',
                                'dst_rf_avg_pr'])

    i = 0 
    for run in runs:
    
        try:
            run_name = open(f'{path}/{run}/params/run_name').read()
        except:
            run_name = "Nan"
        try: 
            experiment_name = open(f'{path}/{run}/params/experiment_name').read()
        except:
            experiment_name = "Nan"
        try:
            manual_seed = open(f'{path}/{run}/params/manual_seed').read()
        except:
            manual_seed = "Nan"



        # first scores that always appear
        try: 
            num_active_units = open(f'{path}/{run}/metrics/num_active_units').read().split()[1]
        except:
            num_active_units = 0.0
        try: 
            gaussian_total_correlation = open(f'{path}/{run}/metrics/gaussian_total_correlation').read().split()[1]
        except:
            gaussian_total_correlation = 0.0
        try:
            gaussian_wasserstein_correlation = open(f'{path}/{run}/metrics/gaussian_wasserstein_correlation').read().split()[1]
        except:
            gaussian_wassertein_correlation = 0.0
        try:
            gaussian_wasserstein_correlation_norm = open(f'{path}/{run}/metrics/gaussian_wasserstein_correlation_norm').read().split()[1]
        except:
            gaussian_wasserstein_correlation_norm = 0.0 

        try: 
            avg_test_loss = open(f'{path}/{run}/metrics/avg_test_loss').read().split()[1]
        except:
            avg_test_loss = 0.0 
        try:
            val_avg_loss = open(f'{path}/{run}/metrics/val_avg_loss').read().split()[1]
        except:
            val_avg_loss = 0.0

        # 1.) color group
        try:
            rf_acc_color_group = open(f'{path}/{run}/metrics/rf_acc_color_group').read().split()[1]
        except:
            rf_acc_color_group = 0.0
        try:
            rf_auc_color_group = open(f'{path}/{run}/metrics/rf_auc_color_group').read().split()[1]
        except:
            rf_auc_color_group = 0.0   
        try:
            rf_aupr_color_group = open(f'{path}/{run}/metrics/rf_aupr_color_group').read().split()[1]
        except:
            rf_aupr_color_group = 0.0
        try:
            rf_avg_pr_color_group = open(f'{path}/{run}/metrics/rf_avg_pr_color_group').read().split()[1]
        except:
            rf_avg_pr_color_group = 0.0 
        try:
            knn_acc_color_group = open(f'{path}/{run}/metrics/knn_acc_color_group').read().split()[1]
        except:
            knn_acc_color_group = 0.0

        # 2.) product_group
        try:
            rf_acc_product_group = open(f'{path}/{run}/metrics/rf_acc_product_group').read().split()[1]
        except:
            rf_acc_product_group = 0.0
        try:
            rf_auc_product_group = open(f'{path}/{run}/metrics/rf_auc_product_group').read().split()[1]
        except:
            rf_auc_product_group = 0.0   
        try:
            rf_aupr_product_group = open(f'{path}/{run}/metrics/rf_aupr_product_group').read().split()[1]
        except:
            rf_aupr_product_group = 0.0
        try:
            rf_avg_pr_product_group = open(f'{path}/{run}/metrics/rf_avg_pr_product_group').read().split()[1]
        except:
            rf_avg_pr_product_group = 0.0 
        try:
            knn_acc_product_group = open(f'{path}/{run}/metrics/knn_acc_product_group').read().split()[1]
        except:
            knn_acc_product_group = 0.0
            
        # 3.) product_type
        try:
            rf_acc_product_type = open(f'{path}/{run}/metrics/rf_acc_product_type').read().split()[1]
        except:
            rf_acc_product_type = 0.0
        try:
            rf_auc_product_type = open(f'{path}/{run}/metrics/rf_auc_product_type').read().split()[1]
        except:
            rf_auc_product_type = 0.0   
        try:
            rf_aupr_product_type = open(f'{path}/{run}/metrics/rf_aupr_product_type').read().split()[1]
        except:
            rf_aupr_product_type = 0.0
        try:
            rf_avg_pr_product_type = open(f'{path}/{run}/metrics/rf_avg_pr_product_type').read().split()[1]
        except:
            rf_avg_pr_product_type = 0.0 
        try:
            knn_acc_product_type = open(f'{path}/{run}/metrics/knn_acc_product_type').read().split()[1]
        except:
            knn_acc_product_type = 0.0

        # 4.) gender
        try:
            rf_acc_gender = open(f'{path}/{run}/metrics/rf_acc_gender').read().split()[1]
        except:
            rf_acc_gender = 0.0
        try:
            rf_auc_gender = open(f'{path}/{run}/metrics/rf_auc_gender').read().split()[1]
        except:
            rf_auc_gender = 0.0   
        try:
            rf_aupr_gender = open(f'{path}/{run}/metrics/rf_aupr_gender').read().split()[1]
        except:
            rf_aupr_gender = 0.0
        try:
            rf_avg_pr_gender = open(f'{path}/{run}/metrics/rf_avg_pr_gender').read().split()[1]
        except:
            rf_avg_pr_gender = 0.0 
        try:
            knn_acc_gender = open(f'{path}/{run}/metrics/knn_acc_gender').read().split()[1]
        except:
            knn_acc_gender = 0.0



        # 5.) age_group
        try:
            rf_acc_age_group = open(f'{path}/{run}/metrics/rf_acc_age_group').read().split()[1]
        except:
            rf_acc_age_group = 0.0
        try:
            rf_auc_age_group = open(f'{path}/{run}/metrics/rf_auc_age_group').read().split()[1]
        except:
            rf_auc_age_group = 0.0   
        try:
            rf_aupr_age_group = open(f'{path}/{run}/metrics/rf_aupr_age_group').read().split()[1]
        except:
            rf_aupr_age_group = 0.0
        try:
            rf_avg_pr_age_group = open(f'{path}/{run}/metrics/rf_avg_pr_age_group').read().split()[1]
        except:
            rf_avg_pr_age_group = 0.0 
        try:
            knn_acc_age_group = open(f'{path}/{run}/metrics/knn_acc_age_group').read().split()[1]
        except:
            knn_acc_age_group = 0.0

        # 6.) if no group specified
        try:
            dst_rf_acc = open(f'{path}/{run}/metrics/dst_rf_acc').read().split()[1]
        except:
            dst_rf_acc = 0.0
        try:
            dst_rf_auc = open(f'{path}/{run}/metrics/dst_rf_auc').read().split()[1]
        except:
            dst_rf_auc = 0.0
        try:
            dst_rf_aupr = open(f'{path}/{run}/metrics/dst_rf_aupr').read().split()[1]
        except:
            dst_rf_aupr = 0.0
        try:
            dst_rf_avg_pr = open(f'{path}/{run}/metrics/dst_rf_avg_pr').read().split()[1]
        except:
            dst_rf_avg_pr = 0.0

        frame.loc[i] = [run_name,
                        experiment_name,
                        manual_seed,
                        num_active_units,
                        gaussian_total_correlation, 
                        gaussian_wasserstein_correlation, 
                        gaussian_wasserstein_correlation_norm, 
                        avg_test_loss,
                        rf_acc_color_group,
                        rf_auc_color_group,
                        rf_aupr_color_group,
                        rf_avg_pr_color_group,
                        knn_acc_color_group,
                        rf_acc_product_group,
                        rf_auc_product_group,
                        rf_aupr_product_group,
                        rf_avg_pr_product_group,
                        knn_acc_product_group,
                        rf_acc_product_type,
                        rf_auc_product_type,
                        rf_aupr_product_type,
                        rf_avg_pr_product_type,
                        knn_acc_product_type,
                        rf_acc_gender,
                        rf_auc_gender,
                        rf_aupr_gender,
                        rf_avg_pr_gender,
                        knn_acc_gender,
                        rf_acc_age_group,
                        rf_auc_age_group,
                        rf_aupr_age_group,
                        rf_avg_pr_age_group,
                        knn_acc_age_group,
                        dst_rf_acc,
                        dst_rf_auc,
                        dst_rf_aupr,
                        dst_rf_avg_pr
                        ]
        i += 1 
    
    frame.to_csv(f'{experiment_name}_runs.csv')



def plot_boxes(df, path, metrics=['rf_acc_product_group', 
                                'rf_acc_color_group',
                                'rf_acc_product_type',
                                'rf_acc_gender']):
    """
    """

    seeds = list(df['manual_seed'].unique())
    for metric in metrics:
        plt.close()
        collections = []
        #seeds = df['manual_seed'].unique()
        collections = [df[df['manual_seed'] == seed][metric] for seed in seeds]
        

        fig = plt.figure(1, figsize=(9, 6))
        ax = fig.add_subplot(111)
        bp = ax.boxplot(collections)

        ax.set_xticklabels(seeds)
        ax.set_xlabel("seeds")
        ax.set_title(f"Seed analysis: {metric}")

        fig.savefig(f"{metric}_boxplot.png")
