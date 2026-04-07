import os

import torch
import multiprocessing as mp
# Set start method to 'spawn'
mp.set_start_method('spawn', force=True)
from .dispatch import TaskDispatcher
from .trainingConfig import MultituneConfig,TaskConfig

class Multitune:
    def __init__(self,config:MultituneConfig):
        self.config = config
        self.GPU_PER_MODEL = 2
        self.wandb_api_key = os.environ["WANDB_API_KEY"]
    
    def finetune(self):
        available_gpus = torch.cuda.device_count()//self.GPU_PER_MODEL
        if available_gpus < len(self.config.tasks):
            raise ValueError("Not enough GPUs for all tasks")
        processes = []
        for i,task in enumerate( self.config.tasks ):
            assigned_gpus = [i*self.GPU_PER_MODEL + j for j in range(self.GPU_PER_MODEL)]
            p = mp.Process(target=TaskDispatcher,args=(self.wandb_api_key,self.config.model_id,self.config.lora_config,task,assigned_gpus))
            processes.append(p)
            # Assign GPUs
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, assigned_gpus))
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
                
                