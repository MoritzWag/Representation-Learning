
import pandas as pd
import numpy as np 
import collections, functools, operator
from library.viz_helpers import plot_train_progress

import torch
import pdb 
from torch import optim 
from library.models2.base2 import ReprLearner
from library import utils
from library.utils import permute_dims
import pytorch_lightning as pl 
from torchvision import transforms 
from torch.utils.data import DataLoader
from torch.autograd import Variable

torch.set_default_dtype(torch.float64)

class RlExperiment(pl.LightningModule):

    def __init__(self, 
                model: ReprLearner,
                params, 
                model_hyperparams,
                run_name,
                experiment_name):
        super(RlExperiment, self).__init__()

        self.model = model.float()
        self.model.epoch = self.current_epoch
        self.params = params
        self.model_hyperparams = model_hyperparams
        self.curr_device = None
        self.train_history = pd.DataFrame()
        self.val_history = pd.DataFrame()
        self.mi_train = pd.DataFrame()
        self.mi_val = pd.DataFrame()
        self.test_score = None
        self.run_name = run_name
        self.experiment_name = experiment_name
        self.accum_index = 0
        self.mut_info = []

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx):

        batch_idx = {'batch_idx': batch_idx}
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

        #if optimizer_idx == 0:
        #    return train_loss
        
        #if optimizer_idx == 1:
        #    z = self.model._embed(image.float(), return_latents=True)
        #    D_xz = self.discriminator(image.float(), z.double())
        #    z_perm = permute_dims(z.double())
        #    D_x_z = self.discriminator(image.float(), z_perm.double())

        #    Info_xz = -(D_xz.mean() - torch.exp(D_x_z - 1).mean())
        #    info_loss = Info_xz

        #    #self.mi_train = self.mi_train.append(info_loss, ignore_index=True)

        #    return {'loss': info_loss}

        return train_loss 

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        
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

            val_loss1 = self.model._loss_function(image.float(), attribute, **reconstruction1)
            val_loss2 = self.model._loss_function(image.float(), attribute, **reconstruction2)
            val_loss3 = self.model._loss_function(image.float(), attribute, **reconstruction3)


            val_loss_dict = [val_loss1, val_loss2, val_loss3]
            counter = collections.Counter()
            for d in val_loss_dict:
                counter.update(d)
            
            val_loss = dict(counter)

        except:
            reconstruction = self.forward(image.float())
            self.model.loss_item['recon_image'] = reconstruction
            val_loss = self.model._loss_function(image.float(), **self.model.loss_item)

            #z = self.model._embed(image.float(), return_latents=True)
            #D_xz = self.discriminator(image.float(), z.double())
            #z_perm = permute_dims(z)
            #D_x_z = self.discriminator(image.float(), z_perm.double())

            Info_xz = -(D_xz.mean() - torch.exp(D_x_z - 1).mean())
            info_loss = Info_xz
        
        val_history = pd.DataFrame([[value.cpu().detach().numpy() for value in val_loss.values()]],
                                    columns=[key for key in val_loss.keys()])

        self.accum_index += val_history.shape[0]
        self.val_history = self.val_history.append(val_history, ignore_index=True)
        
        return val_loss

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean().to(torch.double)
        avg_loss = avg_loss.cpu().detach().numpy() + 0


        self.logger.experiment.log_metric(key='val_avg_loss',
                                        value=avg_loss,
                                        run_id=self.logger.run_id)

        image, attribute = self.model.accumulate_batches(self.val_gen)

        self.model._embed(image)
        
        # COMMENTED OUT FOR EASY ACCESS
        # self.model._sample_images(image,
        #                         path=f"images/{self.params['dataset']}/",
        #                         epoch=self.current_epoch,
        #                         run_name=self.run_name)

        # self.model.traversals(epoch=self.current_epoch,
        #                         run_name=self.run_name,
        #                         path=f"images/{self.params['dataset']}/")

        # self.model._cluster(image=image,
        #                     attribute=attribute[:,0],
        #                     path=f"images/{self.params['dataset']}/",
        #                     epoch=self.current_epoch,
        #                     run_name=self.run_name,
        #                     method='umap')
                            
        # self.model._cluster(image=image,
        #                     attribute=attribute[:,0],
        #                     path=f"images/{self.params['dataset']}/",
        #                     epoch=self.current_epoch,
        #                     run_name=self.run_name,
        #                     method='tsne')

        # self.model._cluster_freq(path=f"images/{self.params['dataset']}/",
        #                         epoch=self.current_epoch,
        #                         run_name=self.run_name)

        del image
        del attribute

        mi = self.model.mutual_information(latent_loss=self.val_history.loc[-self.accum_index:,:]['latent_loss'].mean())
        self.mut_info.append(mi.numpy())
        self.accum_index = 0

        #return {'val_loss': avg_loss}  
        return {'log': {'mut_info': mi.float()}}  

    def test_step(self, batch, batch_idx):
        image, attribute = batch
        image, attribute = image.to(self.curr_device), attribute.to(self.curr_device)

        try:
            reconstruction1 = self.forward(image=image.float(), attrs=attribute)
            reconstruction2 = self.forward(image=image.float())
            reconstruction3 = self.forward(attrs=attribute)

            test_loss1 = self.model._loss_function(image.float(), attribute, **reconstruction1)
            test_loss2 = self.model._loss_function(image.float(), attribute, **reconstruction2)
            test_loss3 = self.model._loss_function(image.float(), attribute, **reconstruction3)

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

        # Loss
        avg_test_loss = torch.stack([x['loss'] for x in outputs]).mean().to(torch.double)
        avg_test_loss = avg_test_loss.cpu().detach().numpy() + 0
        ## log everything with mlflow
        self.logger.experiment.log_metric(key='avg_test_loss', 
                                        value=avg_test_loss,
                                        run_id=self.logger.run_id)

        plot_train_progress(self.train_history,
                            storage_path=f"logs/{self.run_name}/{self.params['dataset']}/training/")
        plot_train_progress(self.val_history,
                            storage_path=f"logs/{self.run_name}/{self.params['dataset']}/validation/")
        
        # Evaluation Metrics
        train_features, train_attributes = self.model.accumulate_batches(data=self.train_gen, return_latents=True)
        test_features, test_attributes = self.model.accumulate_batches(data=self.test_gen, return_latents=True)

        train_data = (train_features, train_attributes)
        test_data = (test_features, test_attributes)
        
        self.model._downstream_task(
            train_data=train_data,
            test_data=test_data,
            model='random_forest',
            storage_path=f"images/{self.params['dataset']}/test/{self.run_name}/",
            downstream_task_names=['color_group','product_group','product_type','gender','age_group']
        )

        self.model._downstream_task(
            train_data,
            test_data,
            'knn_classifier',
            storage_path=f"images/{self.params['dataset']}/test/{self.run_name}/",
            downstream_task_names=['color_group','product_group','product_type','gender','age_group']
        )

        self.model.unsupervised_metrics(test_features)

        self.model.log_metrics(storage_path=f"logs/{self.run_name}/{self.params['dataset']}/test/")

        mut_inf = pd.DataFrame(self.mut_info) 
        mut_inf.to_csv(f"logs/{self.run_name}/{self.params['dataset']}/test/mutual_information.csv")

        for key, value in zip(self.model.scores.keys(), self.model.scores.values()):
            self.logger.experiment.log_metric(key=key,
                                            value=value,
                                            run_id=self.logger.run_id)
        
        for _name, _param in zip(self.params.keys(), self.params.values()):
            self.logger.experiment.log_param(key=_name, 
                                            value=_param,
                                            run_id=self.logger.run_id)
        # log hyperparams
        for _name, _param in zip(self.model_hyperparams.keys(), self.model_hyperparams.values()):
            self.logger.experiment.log_param(key=_name,
                                                value=_param,
                                                run_id=self.logger.run_id)
        
        self.logger.experiment.log_param(key='run_name',
                                        value=self.run_name, 
                                        run_id=self.logger.run_id)
        
        del train_data
        del test_data
        ## Visualization

        image, attribute = self.model.accumulate_batches(self.test_gen)
        self.model._embed(image)

        self.model._sample_images(image,
                        path=f"images/{self.params['dataset']}/test/",
                        epoch=1,
                        run_name=self.run_name)
        
        self.model.traversals(epoch=1,
                                run_name=self.run_name,
                                path=f"images/{self.params['dataset']}/test/")

        self.model._cluster(image=image,
                            attribute=attribute[:,0],
                            path=f"images/{self.params['dataset']}/test/",
                            epoch=1,
                            run_name=self.run_name,
                            method='umap')
                            
        self.model._cluster(image=image,
                            attribute=attribute[:,0],
                            path=f"images/{self.params['dataset']}/test/",
                            epoch=1,
                            run_name=self.run_name,
                            method='tsne')

        self.model._cluster_freq(path=f"images/{self.params['dataset']}/test/",
                                epoch=1,
                                run_name=self.run_name)

        
        self.model._corplot(path=f"images/{self.params['dataset']}/test/",
                            epoch=1,
                            run_name=self.run_name)

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
            data_suffix = ['front_view']
        
        if self.params['dataset'] == 'cifar10':
            path = '/home/ubuntu/data/cifar10/'
            data_suffix=None
        
        train_rawdata, val_rawdata = utils.img_to_npy(path=path,
                                            train=True,
                                            val_split_ratio=0.2,
                                            data_suffix=data_suffix)

        train_data = utils.ImageData(rawdata=train_rawdata, transform=transform,
                                    dataset=self.params['dataset'])

        self.train_gen = DataLoader(dataset=train_data,
                                batch_size=self.params['batch_size'],
                                shuffle=True)


        return self.train_gen

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
            data_suffix=['front_view']
        
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
            data_suffix=['front_view']

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
