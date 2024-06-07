"""
Fabric template 목적
최대한 pytorch lightning 과 비슷하게 동작할 수 있도록 수정
"""
import lightning as L
import torch
import torch.nn as nn
import torch.functional as F

import os
from omegaconf import OmegaConf
import argparse
from typing import Union, List, Any, Optional


from example_model import MnistModel, MNISTLightningModule
from base.base_callback import BaseLightningCallback
from base.base_logger import BaseLightningLogger
from base.data_loaders import BaseDataModule


class BaseFabricTemplate():
    def __init__(self, 
                 config,
                 model, 
                 optimizers: Union[Any, torch.optim.Optimizer],
                 train_dataloaders: Union[Any, None], 
                 valid_dataloaders: Optional[Any]=None,
                 test_dataloaders : Optional[Any]=None,
                 lr_schedulers: Optional[Any]=None, 
                 len_epoch=None, 
                 # new config
                 fabric: L.Fabric=None, 
                 log_freq=1):
        
        
        self.fabric = fabric
        self._fabric(model, train_dataloaders, valid_dataloaders, test_dataloaders, optimizers, lr_schedulers)
        self.config = config
        
        # new param control
        self.log_freq = log_freq
        
    def _fabric(self,model, train_dataloaders, valid_dataloaders, test_dataloaders, optimizers, lr_schedulers):
        """
        check Lightning.fabric and if it is, Convert model, optimizer, dataloader to fabric class
        Args:
            fabric (_type_): Lightning.fabric, Check this params that is Lightning.fabric
        """
        self.fabric.launch(**self.config)
        
        self.model, self.optimizers = self.fabric.setup(model, optimizers)
        assert train_dataloaders is None
        self.train_dataloaders = self.fabric.setup_dataloaders(train_dataloaders)
        self.valid_dataloaders = self.fabric.setup_dataloaders(valid_dataloaders) if not isinstance(valid_dataloaders, None) else None
        self.test_dataloaders = self.fabric.setup_dataloaders(test_dataloaders) if not isinstance(test_dataloaders, None) else None
        
        
        
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, (data, target) in enumerate(self.data_loader):
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            self.fabric.backward(loss)
                
            self.optimizer.step()
            if self.lr_scheduler:
                self.lr_scheduler.step()
            
            if batch_idx % self.log_freq == 0: # log 주기별로 
                self.fabric.log("train_loss", loss)
                # self.fabric.log("train_accuracy", acc)
            
            # fabric 에서 logging 은 자동으로 수행되므로 필요 없음
            # logging
            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, target))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        
        
        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target))
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)


# customize callback
class MyCallback(L.pytorch.callbacks.Callback):
    def on_train_start(self, trainer, pl_module):
        print("Training is starting")

    def on_train_end(self, trainer, pl_module):
        print("Training is ending")


def main(config):
    # model
    model = MnistModel()
    loss_fn = nn.CrossEntropyLoss()
    litmodel = MNISTLightningModule(config, model, loss_fn=loss_fn)
    
    # dataloader
    dm = BaseDataModule(config)
    dm.prepare_data()
    dm.setup('fit')

    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    
    # lightning logger
    logger = BaseLightningLogger(config, CURRENT_TIME)
    tensorboard_logger = logger.TensorBoardLogger()
    wandb_logger = logger.WandbLogger()
    
    logger_list = [wandb_logger, tensorboard_logger]
    ## wandb.watch
    wandb_logger.watch(litmodel, **config['Logger']['wandblogger_watch'])
    
    
    # lightning callback
    callback = BaseLightningCallback(config, CURRENT_TIME)
    # checkpoint_callback = callback.ModelCheckpoint()
    # lr_callback = callback.LearningRateMonitor()
    # earlystop_callback = callback.EarlyStopping()
    
    callback_list = [
        callback.ModelCheckpoint(), 
        callback.LearningRateMonitor(), 
        # callback.EarlyStopping()
        ]
    
    trainer = L.pytorch.lightning.Trainer(
        callbacks=callback_list,
        logger=logger_list,
        **config['Trainer']['init']
    )
    
    # 학습
    trainer.fit(litmodel, train_loader, val_loader, **config['Trainer']['fit'])

def train_with_fabric(config):
    # group 기준으로 저장위치를 변경
    # project->group->name
    
    absolute_file_name = f"{CURRENT_TIME}-batchsize-{config['train_parameters']['batch_size']}-lr-{config['train_parameters']['lr']}-seed-{config['etc']['seed']}"
    save_root_path = f"./{config['trainer']['logger']['WandbLogger']['project']}/{config['trainer']['logger']['WandbLogger']['group']}/{config['trainer']['logger']['WandbLogger']['name']}/{absolute_file_name}"
    
    wandb_logger = L.pytorch.loggers.WandbLogger(
    # name=f"{config['wandblogger']['name']}_{absolute_file_name}", 
    save_dir=save_root_path,
    config=config, 
    **config['trainer']['logger']['WandbLogger']
    )
    
    m = load_model(config)
    loss_fn = nn.CrossEntropyLoss()
    train_loader, val_loader = load_dataloader(config)
    
    Litmodel = TemplateLightningModule(config, m, loss_fn)
    optimizers, schedulers = Litmodel.configure_optimizers()
    # init fabric
    config_trainer_init = config['trainer']['init']
    fabric = L.Fabric( 
        accelerator=config_trainer_init['accelerator'], 
        strategy=config_trainer_init['strategy'], 
        devices=config_trainer_init['devices'],
        precision=config.trainer_init['precision'],
        loggers=[wandb_logger],
    )
    
    fabric.launch()
    model, optimizers = fabric.setup(Litmodel, optimizers)
    train_loader, val_loader = fabric.setup_dataloaders([train_loader, val_loader])
    
    
    
    

if __name__ == "__main__":
    import yaml
    import datetime
    
    parser = argparse.ArgumentParser(description='PyTorch Lightning Template')
    parser.add_argument('-c', '--config', default="./lightning_template_config.yaml", type=str,
                      help='config file path (default: None)')

    args = parser.parse_args()
    config_path = args.config
    
    # config = OmegaConf.load(config_path)
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    CURRENT_TIME = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")  # 굿
    
    
    # sweep_config = {
    #     'method': 'random',
    #     'name': 'first_sweep',
    #     'metric': {
    #         'goal': 'minimize',
    #         'name': 'val_loss'
    #     },
    #     'parameters': {
    #         # 'n_hidden': {'values': [2,3,5,10]},
    #         'lr': {'max': 1.0, 'min': 0.0001},
    #         # 'noise': {'max': 1.0, 'min': 0.}
    #     }
    # }

    # sweep_id=wandb.sweep(sweep_config, project="test_sweep")
    # wandb.agent(sweep_id=sweep_id, function=train, count=5) 
    main(config)