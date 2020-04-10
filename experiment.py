import torch
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

    def forward(self, input, **kwargs):
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx):
        X, Y = batch
    
        reconstruction = self.forward(X.float())
        self.model.loss_item['recon_x'] = reconstruction

        train_loss = self.model._loss_function(X.float(), **self.loss_item)
        loss = train_loss['loss']

        self.logger.experiment.log({key: val.item() for key, val in train_loss.items()})
        
        return train_loss

    def validation_step(self, batch, batch_idx):
        X, Y = batch
        reconstruction = self.forward(X.float())
        self.model.loss_item['recon_x'] = reconstruction
        val_loss = self.model._loss_function(X.float(), **self.model.loss_item)

        return val_loss

    def validation_end(self):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'avg_val_loss': avg_loss}

        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        
        optimizer = optim.Adam(self.model.parameters(),
                                lr=self.params['learning_rate'],
                                weight_decay=self.params['weight_decay'])
        
        try:
            if self.params['scheduler_gamma'] is not None:
                scheduler = optim.lr_scheduler.ExponentialLR(optimizer,
                                                            gamma=self.params['scheduler_gamma'])
                return optimizer, scheduler
        except:
            return optimizer

    def train_dataloader(self):
        #transform = self.data_transforms()

        if self.params['dataset'] == 'mnist':
            path = '../data/mnist/'
        
        train_rawdata, self.val_rawdata = utils.img_to_npy(path=path,
                                            train=True,
                                            val_split_ratio=0.2)

        train_data = utils.ImageData(rawdata=train_rawdata)

        train_gen = DataLoader(dataset=train_data,
                                    batch_size=self.params['batch_size'],
                                    shuffle=True)


        return train_gen

    def val_dataloader(self):

        #transform = self.data_transforms()

        val_data = utils.ImageData(rawdata=self.val_rawdata)
        val_gen = DataLoader(dataset=val_data,
                                batch_size=self.params['batch_size'],
                                shuffle=True)

        return val_gen

    def test_dataloader(self):
        pass

    def data_transforms(self):
        pass


