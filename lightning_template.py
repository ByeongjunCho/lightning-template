import lightning as L
import torch
import torch.nn as nn

import os
from omegaconf import OmegaConf
import argparse

from example_model import MnistModel, MNISTLightningModule
from base.base_callback import BaseLightningCallback
from base.base_logger import BaseLightningLogger
from base.data_loaders import BaseDataModule


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
        callback.EarlyStopping()
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