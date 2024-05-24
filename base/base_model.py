import torch.nn as nn
import numpy as np
import lightning as L
from abc import abstractmethod


class BaseModel(nn.Module):
    """
    Base class for all models
    """
    @abstractmethod
    def forward(self, *inputs):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

class BaseLightningModel(L.LightningModule):
    """
    Base class for LightningModule
    """
    
    @abstractmethod
    def training_step(self):
        raise NotImplementedError
    
    @abstractmethod
    def validation_step(self):
        raise NotImplementedError
    
    @abstractmethod
    def configure_optimizers(self):
        raise NotImplementedError