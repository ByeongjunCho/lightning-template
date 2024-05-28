import lightning as L

import os
import datetime

# 현재 시간 저장(파일 저장 시 절대시간으로 사용)
class BaseLightningCallback():
    def __init__(self, config, CURRENT_TIME):
        self.config = config
        self.callback_config = config['Callback']
        train_batch_size = config['LightningDataModule']['train']['batch_size']
        learning_rate = config['LightningModule']['configure_optimizers']['learning_rate']
        absolute_file_name = f"{CURRENT_TIME}-batchsize-{train_batch_size}-lr-{learning_rate}-seed-{config['etc']['seed']}"
            
        # wandblogger
        project_name = config['Logger']['WandbLogger']['project']
        group_name = config['Logger']['WandbLogger']['group']
        name = config['Logger']['WandbLogger']['name']
        self.save_root_path = f"./{project_name}/{group_name}/{name}/{absolute_file_name}"
        
        # resume : 학습을 이어서 진행하는 경우 연속성을 위해 수정
        resume_path = config['Trainer']['fit']['ckpt_path']
        if resume_path is not None:
            assert resume_path # 파일 있는지 확인
            # wandb resume
            config['Logger']['WandbLogger']['resume'] = 'must'
            # wandb id
            config['Logger']['WandbLogger']['id'] = [x for x in os.listdir(os.path.join('/'.join(resume_path.split('/')[:-2]), 'wandb', 'latest-run')) if 'run-' in x][0].split('-')[-1].replace('.wandb', '')
            # 기존 저장 위치로 변경
            absolute_file_name = resume_path.split('/')[-3]
            self.save_root_path = '/'.join(resume_path.split('/')[:-2])
        
        os.makedirs(self.save_root_path, exist_ok=True)
        
    def ModelCheckpoint(self):
        return L.pytorch.callbacks.ModelCheckpoint(
        filename='{epoch:05d}-{step}-{val_loss:.5f}-{val_acc:.3f}.model',
        dirpath=self.save_root_path + f'/ckpt/', # checkpoint_path
        **self.callback_config['ModelCheckpoint']
    )
        
    def LearningRateMonitor(self):
        return L.pytorch.callbacks.LearningRateMonitor(
        **self.callback_config['LearningRateMonitor']
        )
        
    def EarlyStopping(self):
        return L.pytorch.callbacks.EarlyStopping(
        **self.callback_config['EarlyStopping']
        # monitor='val_loss',
        # min_delta=0,
        # patience=20,
        # verbose=False,
        # mode='min'
        )