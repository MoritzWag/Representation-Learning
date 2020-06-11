
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
                params, 
                experiment_name):
        super(RlExperiment, self).__init__()

        self.model = model.float()
        self.model.epoch = self.current_epoch
        self.params = params
        self.curr_device = None
        self.train_history = pd.DataFrame()
        self.val_history = pd.DataFrame()
        self.test_score = None
        self.experiment_name = experiment_name
        
        #self.logger.experiment.log_param()
        #self.logger.experiment.log_hyperparams(self.params)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx=0):

        batch_idx = {'batch_idx': batch_idx}
        image, attribute = batch
        self.curr_device = image.device
        image, attribute = Variable(image), Variable(attribute)

        try:
            ## here, SOLD_QTY must be deleted 
            reconstruction1 = self.forward(image=image.float(), attrs=attribute)
            reconstruction2 = self.forward(image=image.float())
            reconstruction3 = self.forward(attrs=attribute)

            train_loss1 = self.model._loss_function(image.float(), attribute, **batch_idx, **reconstruction1)
            train_loss2 = self.model._loss_function(image.float(), attribute, **batch_idx, **reconstruction2)
            train_loss3 = self.model._loss_function(image.float(), attribute, **batch_idx, **reconstruction3)
            
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
        #pdb.set_trace()

        batch_idx = {'batch_idx': batch_idx}
        image, attribute = batch
        self.curr_device = image.device
        image, attribute = Variable(image), Variable(attribute)
        
        if torch.cuda.is_available():
            image, attribute = image.cuda(), attribute.cuda()

        try:
            reconstruction1 = self.forward(image=image.float(), attrs=attribute)
            reconstruction2 = self.forward(image=image.float())
            reconstruction3 = self.forward(attrs=attribute)

            val_loss1 = self.model._loss_function(image, attribute, **batch_idx, **reconstruction1)
            val_loss2 = self.model._loss_function(image, attribute, **batch_idx, **reconstruction2)
            val_loss3 = self.model._loss_function(image, attribute, **batch_idx, **reconstruction3)


            val_loss_dict = [val_loss1, val_loss2, val_loss3]
            counter = collections.Counter()
            for d in val_loss_dict:
                counter.update(d)
            
            val_loss = dict(counter)

        except:
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
                                path=f"images/{self.params['dataset']}/",
                                epoch=self.current_epoch,
                                experiment_name=self.experiment_name)
    
        self.model.traversals(data=self.val_gen,
                                epoch=self.current_epoch,
                                experiment_name=self.experiment_name,
                                path=f"images/{self.params['dataset']}/")

        self.model._cluster(data=self.val_gen,
                            path=f"images/{self.params['dataset']}/",
                            epoch=self.current_epoch,
                            experiment_name=self.experiment_name)

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

        plot_train_progress(self.train_history,
                            storage_path=f"logs/{self.experiment_name}/{self.params['dataset']}/training/")
        plot_train_progress(self.val_history,
                            storage_path=f"logs/{self.experiment_name}/{self.params['dataset']}/validation/")

        self.model._downstream_task(self.train_gen, self.test_gen, ...)
        self.model.unsupervised_metrics(self.test_gen, ...)
        self.log_metrics()

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
            data_suffix=None
        
        if self.params['dataset'] == 'fashionmnist':
            path = 'data/fashionmnist/'
            data_suffix=None
        
        if self.params['dataset'] == 'adidas':
            path = '/home/ubuntu/data/adidas/Data/'
            data_suffix = ['standard_view']
        
        if self.params['dataset'] == 'cifar10':
            path = '/home/ubuntu/data/cifar10/'
            data_suffix=None
        
        train_rawdata, val_rawdata = utils.img_to_npy(path=path,
                                            train=True,
                                            val_split_ratio=0.2,
                                            data_suffix=data_suffix)

        train_data = utils.ImageData(rawdata=train_rawdata, transform=transform,
                                    dataset=self.params['dataset'])

        train_gen = DataLoader(dataset=train_data,
                                batch_size=self.params['batch_size'],
                                shuffle=True)


        return train_gen

    def val_dataloader(self):

        transform = self.data_transforms()
        if self.params['dataset'] == 'mnist':
            path = 'data/mnist/'
            data_suffix=None
        
        if self.params['dataset'] == 'fashionmnist':
            path = 'data/fashionmnist/'
            data_suffix=None
        
        if self.params['dataset'] == 'adidas':
            path = '/home/ubuntu/data/adidas/Data/'
            data_suffix=['standard_view']
        
        if self.params['dataset'] == 'cifar10':
            path = '/home/ubuntu/data/cifar10/'
            data_suffix=None

        #transform = self.data_transforms()
        _, val_rawdata = utils.img_to_npy(path=path,
                                        train=True,
                                        val_split_ratio=0.2,
                                        data_suffix=data_suffix)
        

        val_data = utils.ImageData(rawdata=val_rawdata, transform=transform,
                                    dataset=self.params['dataset'])
        self.val_gen = DataLoader(dataset=val_data,
                            batch_size=self.params['batch_size'],
                            shuffle=False)

        return self.val_gen

    def test_dataloader(self):

        transform = self.data_transforms()
        if self.params['dataset'] == 'mnist':
            path = 'data/mnist/'
            data_suffix=None
        
        if self.params['dataset'] == 'fashionmnist':
            path = 'data/fashionmnist/'
            data_suffix=None

        if self.params['dataset'] == 'adidas':
            path = '/home/ubuntu/data/adidas/Data/'
            data_suffix=['standard_view']

        if self.params['dataset'] == 'cifar10':
            path = '/home/ubuntu/data/cifar10/'
            data_suffix=None
        

        test_rawdata = utils.img_to_npy(path=path,
                                        train=False,
                                        data_suffix=data_suffix)
        test_data = utils.ImageData(rawdata=test_rawdata, transform=transform,
                                    dataset=self.params['dataset'])
        self.test_gen = DataLoader(dataset=test_data,
                            batch_size=self.params['batch_size'],
                            shuffle=False)

        return self.test_gen

    def data_transforms(self):
        
        SetRange = transforms.Lambda(lambda X: 2 * X - 1.)
        SetScale = transforms.Lambda(lambda X: X/X.sum(0).expand_as(X))

        if self.params['dataset'] == 'test':
            transform = transforms.Compose([transforms.ToPILImage(),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.CenterCrop(148),
                                            transforms.Resize(224),
                                            transforms.ToTensor(),
                                            SetRange])
        elif self.params['dataset'] == 'test1':
            transform = transforms.Compose([transforms.ToPILImage(),
                                            transforms.CenterCrop(28),
                                            transforms.ToTensor()])
        
        else:
            return None
            
        return transform
