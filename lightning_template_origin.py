import lightning as L
import torch
import torch.nn as nn

import os
from omegaconf import OmegaConf
import argparse

from data_loader.data_loaders import BaseDataModule
from model.model import load_model, TemplateLightningModule

# from wandb.integration.lightning.fabric import WandbLogger
# from lightning.fabric.loggers import CSVLogger, TensorBoardLogger

#########
# 



def train_with_lightning():
    # group 기준으로 저장위치를 변경
    # project->group->name
    
    absolute_file_name = f"{CURRENT_TIME}-batchsize-{config['train_parameters']['batch_size']}-lr-{config['train_parameters']['lr']}-seed-{config['etc']['seed']}"
    save_root_path = f"./{config['trainer']['logger']['WandbLogger']['project']}/{config['trainer']['logger']['WandbLogger']['group']}/{config['trainer']['logger']['WandbLogger']['name']}/{absolute_file_name}"
    
    if config['path']['resume'] is not None:
        absolute_file_name = config['path']['resume'].split('/')[-1]
        save_root_path = config['path']['resume']
    
    
    os.makedirs(save_root_path, exist_ok=True)
    
    # customize modelcheckpoint
    
    # class CustomModelCheckpiont(L.pytorch.callbacks.ModelCheckpoint):
    #     def _save_checkpoint(self, trainer, filepath):
    #         super()._save_checkpoint(trainer, filepath)
    
    # checkpoint_callback = CustomModelCheckpiont(
    #     filename='{epoch:05d}-{step}-{val_loss:.5f}-{val_acc:.3f}.model',
    #     dirpath=save_root_path + f'/ckpt/', # checkpoint_path
    #     **config['trainer']['callback']['ModelCheckpoint']
    # )
    
    checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
        filename='{epoch:05d}-{step}-{val_loss:.5f}-{val_acc:.3f}.model',
        dirpath=save_root_path + f'/ckpt/', # checkpoint_path
        **config['trainer']['callback']['ModelCheckpoint']
    )

    # learningrate callback
    lr_callback = L.pytorch.callbacks.LearningRateMonitor(
        **config['trainer']['callback']['LearningRateMonitor']
        # logging_interval='step',
        # log_momentum=True
    )

    early_stop_callback = L.pytorch.callbacks.EarlyStopping(
        **config['trainer']['callback']['EarlyStopping']
        # monitor='val_loss',
        # min_delta=0,
        # patience=20,
        # verbose=False,
        # mode='min'
        )
    
    # wandb logger
    wandb_logger = L.pytorch.loggers.WandbLogger(
        # name=f"{config['wandblogger']['name']}_{absolute_file_name}", 
        save_dir=save_root_path,
        config=config, 
        **config['trainer']['logger']['WandbLogger']
    )

    
    # tensorboard logger
    tensorboard_logger = L.pytorch.loggers.TensorBoardLogger(save_dir=save_root_path + "/tb_logs",
                                                      **config['trainer']['logger']['TensorBoardLogger']
                                                      )
    
    
    
    model = load_model(config)
    loss_fn = torch.nn.CrossEntropyLoss()
    lit_model = TemplateLightningModule(config, model, loss_fn)
    
    dm = BaseDataModule(config)
    dm.prepare_data()
    dm.setup('fit')
    
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()

    # MPS backend currently does not support all operations used in this example.
    # If you want to use MPS, set accelerator='auto' and also set PYTORCH_ENABLE_MPS_FALLBACK=1
    # accelerator = "cpu" if torch.backends.mps.is_available() else "auto"
    accelerator = "mps" if torch.backends.mps.is_available() else "auto"
    wandb_logger.watch(lit_model, **config['trainer']['logger']['wandblogger_watch'])
    
    trainer = L.pytorch.lightning.Trainer(
        callbacks=[checkpoint_callback, lr_callback],
        # callbacks=[lr_callback],
        logger=[wandb_logger, tensorboard_logger],
        **config['trainer']['init']
    )
    
    trainer.fit(lit_model, train_loader, val_loader, **config['trainer']['fit'])
    
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
    
    # resume 확인
    if config['path']['resume'] is not None:
        assert os.path.exists(config['path']['resume'])
        # ckpt path
        config['trainer']['fit']['ckpt_path'] = os.path.join(config['path']['resume'], 'ckpt', 'last.ckpt')
        # wandb resume
        config['trainer']['logger']['WandbLogger']['resume'] = 'must'
        # wandb id
        config['trainer']['logger']['WandbLogger']['id'] = [x for x in os.listdir(os.path.join(config['path']['resume'], 'wandb')) if 'run-' in x][0].split('-')[-1]

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
    train_with_lightning()