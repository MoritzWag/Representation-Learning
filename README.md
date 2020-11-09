# Representation-Learning
Representation Learning of Image Data with VAE.
Alexander Piehler, Moritz Wagner

## Introduction
This Github Repository is developed by Moritz Wagner and Alexander Piehler in collaboration with Adidas.
The main objective is to learn representation of image and text data with Variational Autoencoders. This codebase allows to run experiments in a reproducible manner and to track experiments via mlflow.
We use PEP8 as styleguide ...

## Code Structure
This framework is structured as follows:
```bash
├── configs
├── data
├── experiments
├── library
├── playground
```

### `configs`
Contains the config files for each model depending on the chosen data set.

### `data`
In this folder, you find all relevant python scripts to download and preprocess the data for running the models.

### `experiments`
Contains the scripts for running all relevant experiments. To run them in the right order, follow these steps:
To run them first change into the directory accordingly: `cd experiments`
1. `python seed_running.py`
2. `python latent_experiment.py`
3. `python epochs_experiment.py`
4. `python kld_weight_experiment.py`
5. `python latent_experiment.py`
6. `python tune_models.py`
7. `python run_best_configurations.py`
8. `python deep_dive.py`

### `library`

This is the package for this repository, it stores all functions and (model) classes that are used. 
The package contains different modules

```bash
├── models2
├── architectures.py
├── eval_helpers.py
├── evaluator.py
├── postprocessing.py
├── utils.py
├── visualizer.py
└── viz_helpers.py
```




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
7. Run code from `run.py --config/config_file.yaml`

# Further notes on running experiments.
In case that hyperparameters want to be adjusted, you can do so by respectively adjusting the parameters set in the `config` files.
Note, however, the parameter configurations listed we found worked best for the respective problem. 
If you want to run your own experiments, it is advisable to additionally specify the `run_name` and the `experiment_name`. 
This is required for `mlflow` to log adequately. 
An exemplary command could be as follows:
`python run.py --configs/MNIST/vae.yaml --run_name mnist_vae --experiment_name mnist_vae`


