import matplotlib 
import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np 
import pdb 
import os
import cv2




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
    #path = f"../../mlruns/{mlflow_id}"
    path = f"mlruns/{mlflow_id}"


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
        model = open(f'{path}{run}/params/model').read()
        
        


    #i = 0 
    #for run in runs:
    #    model = open()




def sort_list_by_other(to_sort, other, reverse=True):
    """Sort a list by an other."""
    return [el for _, el in sorted(zip(other, to_sort), reverse=reverse)]

def get_coordinates(image, dimred_x, dimred_y, global_size):
    # changed from https://www.learnopencv.com/t-sne-for-feature-visualization/
    # Get height and width of image

    height, width, _ = image.shape

    # compute the image center coordinates for dimensionality reduction
    # plot
    center_x = int(global_size * dimred_x)

    # to have the same here, we need to mirror the y coordinate
    center_y = int(global_size * (1 - dimred_y))

    # Compute edge coordinates
    topleft_x = center_x - int(width / 2)
    topleft_y = center_y - int(height / 2)

    bottomright_x = center_x + int(width / 2)
    bottomright_y = center_y + int(height / 2)

    if topleft_x < 0:
        bottomright_x = bottomright_x + abs(topleft_x)
        topleft_x = topleft_x + abs(topleft_x)

    if topleft_y < 0:
        bottomright_y = bottomright_y + abs(topleft_y)
        topleft_y = topleft_y + abs(topleft_y)

    if bottomright_x > global_size:
        topleft_x = topleft_x - (bottomright_x - global_size)
        bottomright_x = bottomright_x - (bottomright_x - global_size)

    if bottomright_y > global_size:
        topleft_y = topleft_y - (bottomright_y - global_size)
        bottomright_y = bottomright_y - (bottomright_y - global_size)

    return topleft_x, topleft_y, bottomright_x, bottomright_y

def reshape_image(img, scaling):
    """
    Input: NumpyArray {[H,W,C]}
    Output: Resized numpy array {[H,W,C]}
    """
    # Undo scaling
    img = img * 255

    width = int(img.shape[1] * (scaling / 100))
    height = int(img.shape[0] * (scaling / 100))
    dim = (width, height)

    # resize image
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    return resized