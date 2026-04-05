from typing import Any

from attr import dataclass
from datasets import Dataset
from peft import LoraConfig
import wandb
import torch
import multiprocessing as mp
from src.dispatch import TaskDispatcher
wandb.login()

@dataclass
class TaskConfig:
    name: str
    dataset: Dataset
    trainer_config: Any

@dataclass
class MultituneConfig:
    model_id: str
    lora_config: LoraConfig
    tasks: list[TaskConfig]

class Multitune:
    def __init__(self,config:MultituneConfig):
        self.config = config
        self.GPU_PER_MODEL = 1
    
    def finetune(self):
        available_gpus = torch.cuda.device_count()//self.GPU_PER_MODEL
        if available_gpus < len(self.config.tasks):
            raise ValueError("Not enough GPUs for all tasks")
        processes = []
        for i,task in enumerate( self.config.tasks ):
            assigned_gpus = [i*self.GPU_PER_MODEL + j for j in range(self.GPU_PER_MODEL)]
            p = mp.Process(target=TaskDispatcher,args=(self.config.model_id,self.config.lora_config,task,assigned_gpus))
            processes.append(p)
            p.start()
        try:
            for p in processes:
                p.join()
        except KeyboardInterrupt:
            print("Stopping workers...")
            for p in processes:
                p.terminate()
            for p in processes:
                p.join(timeout=3)
            for p in processes:
                if p.is_alive():
                    p.kill()
                
                