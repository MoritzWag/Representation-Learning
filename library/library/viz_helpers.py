import matplotlib 
import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np 
import pdb 
import os



def plot_train_progress(history,
                        storage_path,
                        ):
    """plot training progress based on
    training/validation history.

    """
    num_metrics = len(history.columns)
    plt.close()
    plt.figure(figsize=(20, 12))
    for metric in range(num_metrics):
        plt.subplot(num_metrics, 1, metric + 1)
        plt.plot(history.iloc[20:, metric])
        plt.xlabel('training steps')
        plt.ylabel(history.columns[metric])

    plt.tight_layout()

    if storage_path is not None:
        if not os.path.exists(storage_path):
            os.makedirs(f"./{storage_path}")
        plt.savefig(f"{storage_path}.png")



def get_mlflow_results(mlflow_id):
    path = f"../../mlruns{mlflow_id}"

    # select only runy with 32-lenghts hashes
    runs = [run for run in os.listdir(path) if len(run) == 32 and not run.startswith('performance')]
    frame = pd.DataFrame(columns=['run_id', 'model', 'num_epochs', 'dataset', 
                                'avg_test_loss', 'val_avg_loss', 'mutual_info_score',
                                'gaussian_total_correlation', 'gaussian_wassserstein_correlation',
                                'gaussian_wasserstein_correlation_norm',
                                'downstream_task_acc1', 'downstream_task_acc2',
                                'downstream_task_acc3', 'downstream_task_acc4',
                                'downstream_task_acc5',
                                ])
    
    i = 0 
    for run in runs:
        dataset = open(f'{path}{run}/params/dataset').read()
        


    #i = 0 
    #for run in runs:
    #    model = open()




def sort_list_by_other(to_sort, other, reverse=True):
    """Sort a list by an other."""
    return [el for _, el in sorted(zip(other, to_sort), reverse=reverse)]

