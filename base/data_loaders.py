import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torchvision import transforms

import lightning as L


# base datamodule(can be customized)
class BaseDataModule(L.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.train_config = config["LightningDataModule"]['train']
        self.val_config = config["LightningDataModule"]['val']
        self.test_config = config["LightningDataModule"]['test']
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])        
    def prepare_data(self):
        """
        single cpu 에서 한번만 실행되므로 아래 작업에 용이함
        - download
        - tokenize(1 processor 에서만 실행되므로 잘 고려해서 사용)
        - etc
        """
        
        MNIST(root=self.train_config['src_path'], train=True, transform=ToTensor(), download=True)
        MNIST(root=self.val_config['src_path'], train=False, transform=ToTensor(), download=False)
        
    def setup(self, stage: str):
        """
        모든 gpu 에서 실행됨. 아래 예시 수행
        - count number of classes
        - build vocabulary
        - perform train/val/test splits
        - create datasets
        - apply transforms (defined explicitly in your datamodule)
        - etc…
        
        Args:
            stage (str): set fit(train with valid) or test(test) or predicition(prediction)
        """
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            mnist_full = MNIST(self.train_config['src_path'], train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(
                mnist_full, [55000, 5000], generator=torch.Generator().manual_seed(42)
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.mnist_test = MNIST(self.test_config['src_path'], train=False, transform=self.transform)

        if stage == "predict":
            self.mnist_predict = MNIST(self.test_config['src_path'], train=False, transform=self.transform)
    
    def train_dataloader(self):
        return DataLoader(
            self.mnist_train, 
            batch_size=self.train_config['batch_size'],
            shuffle=self.train_config['shuffle'],
            pin_memory=torch.cuda.is_available(), 
            num_workers=self.train_config['num_workers']
            )

    def val_dataloader(self):
        return DataLoader(
            self.mnist_val, 
            batch_size=self.val_config['batch_size'], 
            shuffle=self.val_config['shuffle'],
            pin_memory=torch.cuda.is_available(),
            num_workers=self.val_config['num_workers']
            )

    def test_dataloader(self):
        return DataLoader(
            self.mnist_test, 
            batch_size=self.test_config['batch_size'], 
            pin_memory=torch.cuda.is_available(),
            num_workers=self.test_config['num_workers']
            )

    def predict_dataloader(self):
        return DataLoader(self.mnist_predict, batch_size=32)

if __name__ == "__main__":
    import yaml
    with open("/Users/byeongjuncho/PythonProject/etc/lightning-template/lightning_template_config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    dm = BaseDataModule(config=config)
    dm.prepare_data()
    dm.setup(stage='fit') # train and valid
    
    train_dataloader = dm.train_dataloader()
    val_dataloader = dm.val_dataloader()
    a, b = train_dataloader.__iter__().__next__()
    print(b)
