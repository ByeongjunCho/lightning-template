import lightning as L
import torch
import torch.nn as nn

import wandb

from torchmetrics.functional.classification.accuracy import accuracy

import os
# from wandb.integration.lightning.fabric import WandbLogger
# from lightning.fabric.loggers import CSVLogger, TensorBoardLogger


class TemplateLightningModule(L.LightningModule):
    def __init__(self, config, model, loss_fn) -> None:
        super().__init__()
        self.config = config
        # self.sweep_config = sweep_config
        self.model = model
        self.loss_fn = loss_fn
        self.save_hyperparameters(ignore=['model', 'loss_fn'])
        
    def forward(self, x: torch.Tensor):
        return self.model(x)
    
    def calculate_matric(self, logits, labels):
        loss = self.loss_fn(logits, labels)
        acc = accuracy(logits.argmax(-1), labels, num_classes=10, task="multiclass", top_k=1)
        
        return loss, acc

    def training_step(self, batch, batch_idx: int):
        x, y = batch
        logits = self(x)

        loss, acc = self.calculate_matric(logits, y)
        
        self.log_dict({'train_loss': loss, 'train_acc': acc, 'epoch': self.current_epoch}, sync_dist=True, prog_bar=True)
        
        return {"loss": loss, "accuracy": acc}
    
    def validation_step(self, batch, batch_idx:int):
        x, y = batch
        logits = self(x)
        loss, acc = self.calculate_matric(logits, y)
        
        self.log_dict({'val_loss': loss, 'val_acc': acc, 'epoch': self.current_epoch}, sync_dist=True, prog_bar=True)
        
        return {"loss": loss, "accuracy": acc}
    
    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=float(self.config['train_parameters']['lr']))
        scheduler_config = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode="max", verbose=True),
            "monitor": "val_acc",
            "interval": "epoch",
            "frequency": 1,
        }
        return [optim], [scheduler_config] # multiple optim, multiple scheduler


    

def train():
    absolute_file_name = f"{CURRENT_TIME}-batchsize-{config['train_parameters']['batch_size']}-lr-{config['train_parameters']['lr']}-seed-{config['etc']['seed']}"
    save_root_path = f"./{config['trainer']['logger']['WandbLogger']['project']}/{config['trainer']['logger']['WandbLogger']['group']}/{config['trainer']['logger']['WandbLogger']['name']}/{absolute_file_name}"
    
    if config['path']['resume'] is not None:
        absolute_file_name = config['path']['resume'].split('/')[-1]
        save_root_path = config['path']['resume']
    
    
    os.makedirs(save_root_path, exist_ok=True)
    checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
        filename='{epoch:05d}-{val_loss:.5f}-{val_acc:.3f}.model',
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
    
    
    model = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Conv2d(16, 32, 5, 1, 2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Flatten(),
            # fully connected layer, output 10 classes
            torch.nn.Linear(32 * 7 * 7, 10),
        )
    
    loss_fn = torch.nn.CrossEntropyLoss()
    
    lit_model = TemplateLightningModule(config, model, loss_fn)
    
    # dataset and dataloader load
    from torchvision.datasets import MNIST
    from torchvision.transforms import ToTensor

    train_set = MNIST(root="/tmp/data/MNIST", train=True, transform=ToTensor(), download=True)
    val_set = MNIST(root="/tmp/data/MNIST", train=False, transform=ToTensor(), download=False)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=64, shuffle=True, pin_memory=torch.cuda.is_available(), num_workers=4
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=64, shuffle=False, pin_memory=torch.cuda.is_available(), num_workers=4
    )

    # MPS backend currently does not support all operations used in this example.
    # If you want to use MPS, set accelerator='auto' and also set PYTORCH_ENABLE_MPS_FALLBACK=1
    # accelerator = "cpu" if torch.backends.mps.is_available() else "auto"
    accelerator = "mps" if torch.backends.mps.is_available() else "auto"
    wandb_logger.watch(lit_model, **config['trainer']['logger']['wandblogger_watch'])
    
    trainer = L.pytorch.lightning.Trainer(
        callbacks=[checkpoint_callback, lr_callback],
        logger=[wandb_logger, tensorboard_logger],
        **config['trainer']['init']
    )
    
    trainer.fit(lit_model, train_loader, val_loader, **config['trainer']['fit'])
    

if __name__ == "__main__":
    import yaml
    import datetime
    with open('/Users/byeongjuncho/PythonProject/etc/template/lightning_template_config.yaml', 'r') as f:
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
    train()