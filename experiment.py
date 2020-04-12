
import pandas as pd
import numpy as np 
from library.visualizations import plot_train_progress

import torch
import pdb 
from torch import optim 
from library.models2.base2 import ReprLearner
from library import utils
import pytorch_lightning as pl 
from torchvision import transforms 
from torch.utils.data import DataLoader

class RlExperiment(pl.LightningModule):

    def __init__(self, 
                model: ReprLearner,
                params):
        super(RlExperiment, self).__init__()
        self.model = model.float()
        self.params = params
        self.curr_device = None
        self.train_history = pd.DataFrame()
        self.val_history = pd.DataFrame()
        self.test_score = None
        self.experiment_name = "VaeGaussian"
    
        #self.logger.experiment.log_param()
        #self.logger.experiment.log_hyperparams(self.params)

    def forward(self, input, **kwargs):
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx=0):

        X, Y = batch
    
        reconstruction = self.forward(X.float())
        self.model.loss_item['recon_x'] = reconstruction

        train_loss = self.model._loss_function(X.float(), **self.model.loss_item)
        #for item in train_loss.items():
        #    self.logger.experiment.log_metric(key=item[0], 
        #                                    value=item[1].float(),
        #                                    run_id=self.logger.run_id)
        
        train_history = pd.DataFrame([[value.detach().numpy() for value in train_loss.values()]],
                                    columns=[key for key in train_loss.keys()])
        
        self.train_history = self.train_history.append(train_history, ignore_index=True)

        # here some visualization functionality 

        return train_loss

    def validation_step(self, batch, batch_idx):
        X, Y = batch

        reconstruction = self.forward(X.float())
        self.model.loss_item['recon_x'] = reconstruction
        val_loss = self.model._loss_function(X.float(), **self.model.loss_item)

        #for item in val_loss.items():
        #    self.logger.experiment.log_metric(key=item[0],
        #                                    value=item[1],
        #                                    run_id=self.logger.run_id)
        
        val_history = pd.DataFrame([[value.detach().numpy() for value in val_loss.values()]],
                                    columns=[key for key in val_loss.keys()])

        self.val_history = self.val_history.append(val_history, ignore_index=True)

        # here some visualization functionality 
        # sample and reconstruct images for visualization
        self.model._sample_images(self.val_gen,
                                path='images/',
                                epoch=self.current_epoch,
                                experiment_name='VaeExperiment')

        return val_loss

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean().to(torch.double)
        #self.logger.experiment.log_metric(key='val_avg_loss',
        #                                value=avg_loss,
        #                                run_id=self.logger.run_id)

        return {'val_loss': avg_loss}    

    def test_step(self, batch, batch_idx):
        X, Y = batch
        reconstruction = self.forward(X.float())
        self.model.loss_item['recon_x'] = reconstruction
        test_loss = self.model._loss_function(X.float(), **self.model.loss_item)

        self.test_score = test_loss   

        return test_loss

    def test_epoch_end(self, outputs):
        avg_test_loss = torch.stack([x['loss'] for x in outputs]).mean().to(torch.double)

        ## log everything with mlflow

        # here should be the call of the visualization function
        # for train_history and val_history. 
        plot_train_progress(self.train_history,
                            storage_path=f"logs/{self.experiment_name}/training/")
        plot_train_progress(self.val_history,
                            storage_path=f"logs/{self.experiment_name}/validation/")

        return {'avg_test_loss': avg_test_loss}

    def configure_optimizers(self):

        optims = []
        scheds = []
        optimizer = optim.Adam(self.model.parameters(),
                                lr=self.params['learning_rate'],
                                weight_decay=self.params['weight_decay'])
        optims.append(optimizer)
        try:
            if self.params['scheduler_gamma'] is not None:
                scheduler = optim.lr_scheduler.ExponentialLR(optimizer,
                                                            gamma=self.params['scheduler_gamma'])
                
                scheds.append(scheduler)
                return optims, scheds
        except:
            return optims

    def train_dataloader(self):
        #transform = self.data_transforms()

        if self.params['dataset'] == 'mnist':
            path = 'data/mnist/'
        
        train_rawdata, _ = utils.img_to_npy(path=path,
                                            train=True,
                                            val_split_ratio=0.2)

        train_data = utils.ImageData(rawdata=train_rawdata)

        train_gen = DataLoader(dataset=train_data,
                                batch_size=self.params['batch_size'],
                                shuffle=True)


        return train_gen

    def val_dataloader(self):

        if self.params['dataset'] == 'mnist':
            path = 'data/mnist/'

        #transform = self.data_transforms()
        _, val_rawdata = utils.img_to_npy(path=path,
                                        train=True,
                                        val_split_ratio=0.2)

        val_data = utils.ImageData(rawdata=val_rawdata)
        self.val_gen = DataLoader(dataset=val_data,
                            batch_size=self.params['batch_size'],
                            shuffle=False)

        return self.val_gen

    def test_dataloader(self):
        if self.params['dataset'] == 'mnist':
            path = 'data/mnist/'
        
        test_rawdata = utils.img_to_npy(path=path,
                                        train=False)
        test_data = utils.ImageData(rawdata=test_rawdata)
        test_gen = DataLoader(dataset=test_data,
                            batch_size=self.params['batch_size'],
                            shuffle=False)

        return test_gen

    def data_transforms(self):
        pass


    def any_other_function(self):
        pass