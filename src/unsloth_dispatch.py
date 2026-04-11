from unsloth import FastLanguageModel
import os
from .trainingConfig import TaskConfig
import wandb

from .multitune import Multitune

def UnslothTaskDispatcher(report_key:str,model_id:str,lora_config:dict,task:TaskConfig,assigned_gpus:list[int]):
    # TODO: May drop assigned_gpus?
    if not assigned_gpus:
        raise ValueError("TaskDispatcher requires at least one assigned GPU")
    print("Assigned GPUs:",os.environ["CUDA_VISIBLE_DEVICES"],assigned_gpus)

    data = task.dataset
    data = data.map(task.data_formatter, remove_columns=data.column_names)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name= model_id,
        gpu_memory_utilization = 0.9, # Reduce if out of memory
        # fast_inference = True, # Enable vllm fast inference
        device_map="balanced"
    )
    model = FastLanguageModel.get_peft_model(
        model,
        **lora_config
    )

    trainer = task.trainer_class(
        model=model,
        args=task.trainer_config,
        train_dataset=data,
        **task.trainer_kwargs
    )
    wandb.login(report_key)
    wandb.init(project=task.name,entity=task.entity)
    trainer.train()

class UnslothMultitune(Multitune):
    def get_dispatcher(self):
        return UnslothTaskDispatcher