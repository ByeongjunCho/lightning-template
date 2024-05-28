"""
여기서 pytorch nn.Module 과 LightningModule 선언
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional.classification.accuracy import accuracy
import lightning as L

import transformers

from base.base_model import BaseModel, BaseLightningModel


class MNISTLightningModule(BaseLightningModel):
    def __init__(self, config, model, loss_fn) -> None:
        super().__init__()
        self.config_lightningmodule = config['LightningModule']
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
        optimizer = torch.optim.AdamW(self.parameters(), lr=float(self.config_lightningmodule['configure_optimizers']['learning_rate']))

        # total_steps = self.trainer.estimated_stepping_batches
        # scheduler_config = {
        #     "scheduler": transformers.get_cosine_schedule_with_warmup(
        #         optimizer, 
        #         num_warmup_steps=total_steps//4, 
        #         num_training_steps=total_steps
        #         ),
        #     "monitor": "val_acc",
        #     "interval": "step",
        #     "frequency": 1,
        # }
        
        # AdamW8bit example
        # import bitsandbytes as bnb
        # optim = bnb.optim.AdamW8bit(self.model.model.embed_tokens.parameters(), lr=float(self.config['train_parameters']['lr']), betas=(0.9, 0.995))
        
        # # (optional) force embedding layers to use 32 bit for numerical stability
        # # https://github.com/huggingface/transformers/issues/14819#issuecomment-1003445038
        # for module in self.model.modules():
        #     if isinstance(module, torch.nn.Embedding):
        #         bnb.optim.GlobalOptimManager.get_instance().register_module_override(
        #             module, "weight", {"optim_bits": 32}
        
        return [optimizer] # multiple optim, multiple scheduler


    ### checkpoint 변형 - 아래 함수는 hook 중 저장하는 과정에서 실행되며, 여기서 state_dict 를 컨트롤할 수 있음
    
    # def on_save_checkpoint(self, checkpoint):
    #     checkpoint["state_dict"]
    
    
class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.model = torch.nn.Sequential(
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

    def forward(self, x):
        return self.model(x)

if __name__ == "__main__":
    import yaml
    with open("/Users/byeongjuncho/PythonProject/etc/lightning-template/lightning_template_config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    
    model = MnistModel(num_classes=10)
    print(model)
    
    loss_fn = nn.CrossEntropyLoss()
    litmodel = MNISTLightningModule(config, model, loss_fn=loss_fn)