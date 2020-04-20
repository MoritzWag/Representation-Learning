# Representation-Learning
Representation Learning of Image Data with VAE.

## Introduction
This Github Repository is developed by Moritz Wagner and Alexander Piehler in collaboration with Adidas.
The main objective is to learn representation of image and text data with Variational Autoencoders. This codebase allows to run experiments in a reproducible manner and to track experiments via mlflow.
We use PEP8 as styleguide ...

## Code Structure


## Packaging
* package `library`enables pip installation and symling creation
* fork the repository and execute `pip install -e .` from the parent tree.
* packaging allows also for updating the packages in the scripts via `importlib.reload()`



## Setup
You can setup Representation-Learning as follows:
Note, all commands must be run from parent level of the repository.
1. Install miniconda 
2. Create a conda environmnet for python 3.7 (`conda create -n <env name> python=3.7`)
3. Clone this repository
4. Install the required packages via `pip install -r requirements.txt`
5. Install the local package `library` as described above.
6. Download the data by moving to data folder `cd data` and executing `python get_<dataset>_data.py`
7. Run code from `run.py`

# Further notes on running experiments.
In case that hyperparameters want to be adjusted, you can do so by respectively adjusting the parameters set in the `config` files.
Note, however, the parameter configurations listed we found worked best for the respective problem. 


