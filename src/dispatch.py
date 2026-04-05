from .trainingConfig import TaskConfig
from peft import LoraConfig, PeftModel, get_peft_model
import os

from transformers import AutoModelForCausalLM, AutoTokenizer

def TaskDispatcher(model_id:str,lora_config:LoraConfig,task:TaskConfig,assigned_gpus:list[int]):
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str,assigned_gpus))
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "left"
    raw_model = AutoModelForCausalLM.from_pretrained(model_id)
    model = get_peft_model(raw_model,lora_config)
    trainer = task.trainer_class(
        model=model,
        args=task.trainer_config,
        train_dataset=task.dataset,
        **task.trainer_kwargs
    )
    wandb.init(project=task.name,entity=task.entity)
    trainer.train()
