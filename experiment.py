
import pandas as pd
import numpy as np 
import collections, functools, operator
from library.viz_helpers import plot_train_progress

import torch
import pdb 
from torch import optim 
from library.models2.base2 import ReprLearner
from library import utils
import pytorch_lightning as pl 
from torchvision import transforms 
from torch.utils.data import DataLoader
from torch.autograd import Variable

torch.set_default_dtype(torch.float64)

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

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        image, attribute = batch
        self.curr_device = image.device
        image, attribute = Variable(image), Variable(attribute)

        try:
            reconstruction1 = self.forward(image=image.float(), attrs=attribute)
            reconstruction2 = self.forward(image=image.float())
            reconstruction3 = self.forward(attrs=attribute)

            train_loss1 = self.model._loss_function(image.float(), attribute, **reconstruction1)
            train_loss2 = self.model._loss_function(image.float(), attribute, **reconstruction2)
            train_loss3 = self.model._loss_function(image.float(), attribute, **reconstruction3)
            
            train_loss_dict = [train_loss1, train_loss2, train_loss3]
            counter = collections.Counter()
            for d in train_loss_dict:
                counter.update(d)
            
            train_loss = dict(counter)
            print(train_loss)

        except:
            reconstruction = self.forward(image.float())
            self.model.loss_item['recon_image'] = reconstruction
            train_loss = self.model._loss_function(image.float(), **self.model.loss_item)
        
        train_history = pd.DataFrame([[value.cpu().detach().numpy() for value in train_loss.values()]],
                                    columns=[key for key in train_loss.keys()])   
            
        self.train_history = self.train_history.append(train_history, ignore_index=True)


        return train_loss



    def validation_step(self, batch, batch_idx, optimizer_idx=0):

        image, attribute = batch
        self.curr_device = image.device
        image, attribute = Variable(image), Variable(attribute)
        image, attribute = image.cuda(), attribute.cuda()

        try:
            reconstruction1 = self.forward(image=image.float(), attrs=attribute)
            reconstruction2 = self.forward(image=image.float())
            reconstruction3 = self.forward(attrs=attribute)

            val_loss1 = self.model._loss_function(image, attribute, **reconstruction1)
            val_loss2 = self.model._loss_function(image, attribute, **reconstruction2)
            val_loss3 = self.model._loss_function(image, attribute, **reconstruction3)


            val_loss_dict = [val_loss1, val_loss2, val_loss3]
            counter = collections.Counter()
            for d in val_loss_dict:
                counter.update(d)
            
            val_loss = dict(counter)

        except:
            #pdb.set_trace()
            reconstruction = self.forward(image.float())
            self.model.loss_item['recon_image'] = reconstruction
            val_loss = self.model._loss_function(image.float(), **self.model.loss_item)

        #for item in val_loss.items():
        #    self.logger.experiment.log_metric(key=item[0],
        #                                    value=item[1],
        #                                    run_id=self.logger.run_id)
        
        val_history = pd.DataFrame([[value.cpu().detach().numpy() for value in val_loss.values()]],
                                    columns=[key for key in val_loss.keys()])

        self.val_history = self.val_history.append(val_history, ignore_index=True)

        return val_loss

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean().to(torch.double)
        #self.logger.experiment.log_metric(key='val_avg_loss',
        #                                value=avg_loss,
        #                                run_id=self.logger.run_id)

        self.model._sample_images(self.val_gen,
                                path='images/',
                                epoch=self.current_epoch,
                                experiment_name='VaeExperiment')

        self.model.traversals(data=self.val_gen,
                            is_reorder_latents=False,
                            n_per_latent=8,
                            n_latents=None,
                            epoch=self.current_epoch,
                            experiment_name='VaeExperiment',
                            path='images/')

        self.model._cluster(data=self.val_gen,
                            path='images/',
                            epoch=self.current_epoch,
                            experiment_name='VaeExperiment')

        return {'val_loss': avg_loss}    

    def test_step(self, batch, batch_idx):
        image, attribute = batch
        image, attribute = image.to(self.curr_device), attribute.to(self.curr_device)

        try:
            reconstruction1 = self.forward(image=image.float(), attrs=attribute)
            reconstruction2 = self.forward(image=image.float())
            reconstruction3 = self.forward(attrs=attribute)

            test_loss1 = self.model._loss_function(image, attribute, **reconstruction1)
            test_loss2 = self.model._loss_function(image, attribute, **reconstruction2)
            test_loss3 = self.model._loss_function(image, attribute, **reconstruction3)

            test_loss_dict = [test_loss1, test_loss2, test_loss3]
            counter = collections.Counter()
            for d in test_loss_dict:
                counter.update(d)
            
            test_loss = dict(counter)
            
        except:
            reconstruction = self.forward(image.float())
            self.model.loss_item['recon_image'] = reconstruction
            test_loss = self.model._loss_function(image.float(), **self.model.loss_item)

        return test_loss

    def test_epoch_end(self, outputs):
        avg_test_loss = torch.stack([x['loss'] for x in outputs]).mean().to(torch.double)

        ## log everything with mlflow

        # here should be the call of the visualization function
        # for train_history and val_history. 
        # call logging_params from yaml file
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
                scheduler = optim.lr_scheduler.ExponentialLR(optims[0],
                                                            gamma=self.params['scheduler_gamma'])
                
                scheds.append(scheduler)
                return optims, scheds
        except:
            return optims

    def train_dataloader(self):
        transform = self.data_transforms()

        if self.params['dataset'] == 'mnist':
            path = 'data/mnist/'
        
        if self.params['dataset'] == 'adidas':
            path = '/home/ubuntu/data/adidas/Data/'
        
        #path = f"data/{self.params['dataset']}"
        
        train_rawdata, val_rawdata = utils.img_to_npy(path=path,
                                            train=True,
                                            val_split_ratio=0.2)
        #self.val_rawdata = val_rawdata
        train_data = utils.ImageData(rawdata=train_rawdata, transform=transform)

        train_gen = DataLoader(dataset=train_data,
                                batch_size=self.params['batch_size'],
                                shuffle=True)


        return train_gen

    def val_dataloader(self):

        transform = self.data_transforms()
        if self.params['dataset'] == 'mnist':
            path = 'data/mnist/'
        
        if self.params['dataset'] == 'adidas':
            path = '/home/ubuntu/data/adidas/Data/'
        
        #path = f"data/{self.params['dataset']}"

        #transform = self.data_transforms()
        _, val_rawdata = utils.img_to_npy(path=path,
                                        train=True,
                                        val_split_ratio=0.2)
        

        val_data = utils.ImageData(rawdata=val_rawdata, transform=transform)
        self.val_gen = DataLoader(dataset=val_data,
                            batch_size=self.params['batch_size'],
                            shuffle=False)

        return self.val_gen

    def test_dataloader(self):

        transform = self.data_transforms()
        if self.params['dataset'] == 'mnist':
            path = 'data/mnist/'

        if self.params['dataset'] == 'adidas':
            path = '/home/ubuntu/data/adidas/Data/'
        
        #path = f"data/{self.params['dataset']}"
        
        test_rawdata = utils.img_to_npy(path=path,
                                        train=False)
        test_data = utils.ImageData(rawdata=test_rawdata, transform=transform)
        self.test_gen = DataLoader(dataset=test_data,
                            batch_size=self.params['batch_size'],
                            shuffle=False)

        return self.test_gen

    def data_transforms(self):
        
        SetRange = transforms.Lambda(lambda X: 2 * X - 1.)
        SetScale = transforms.Lambda(lambda X: X/X.sum(0).expand_as(X))

        if self.params['dataset'] == 'adidas':
            transform = transforms.Compose([transforms.ToPILImage(),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.CenterCrop(148),
                                            transforms.Resize(224),
                                            transforms.ToTensor(),
                                            SetRange])
        else:
            return None
            
        return transform
