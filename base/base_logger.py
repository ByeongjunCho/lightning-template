import lightning as L

import os
import datetime

# 현재 시간 저장(파일 저장 시 절대시간으로 사용)
class BaseLightningLogger():
    def __init__(self, config, CURRENT_TIME):
        self.config = config
        self.logger_config = config['Logger']
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
        if config['Trainer']['fit']['ckpt_path'] is not None:
            assert resume_path # 파일 있는지 확인
            # wandb resume
            config['Logger']['WandbLogger']['resume'] = 'must'
            # wandb id
            config['Logger']['WandbLogger']['id'] = [x for x in os.listdir(os.path.join('/'.join(resume_path.split('/')[:-2]), 'wandb')) if 'run-' in x][0].split('-')[-1]
            # 기존 저장 위치로 변경
            absolute_file_name = resume_path.split('/')[-3]
            self.save_root_path = '/'.join(resume_path.split('/')[:-2])
        
        os.makedirs(self.save_root_path, exist_ok=True)
            
    def WandbLogger(self):
        wandb_logger = L.pytorch.loggers.WandbLogger(
            # name=f"{config['wandblogger']['name']}_{absolute_file_name}", 
            save_dir=self.save_root_path,
            config=self.config,
            **self.logger_config['WandbLogger']
            )
        return wandb_logger
        
        
    def TensorBoardLogger(self):
        return L.pytorch.loggers.TensorBoardLogger(
            save_dir=self.save_root_path + "/tb_logs",
            **self.logger_config['TensorBoardLogger']
            )