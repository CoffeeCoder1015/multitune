import os

from .trainingConfig import TaskConfig
from peft import LoraConfig, get_peft_model
import torch
import wandb

from transformers import AutoModelForCausalLM, AutoTokenizer


def build_max_memory(assigned_gpus: list[int], memory_buffer_ratio: float = 0.9) -> dict[int, str]:
    max_memory = {}
    for gpu_id in assigned_gpus:
        total_memory = torch.cuda.get_device_properties(gpu_id).total_memory / (1024**3)
        usable_memory = max(int(total_memory * memory_buffer_ratio), 1)
        max_memory[gpu_id] = f"{usable_memory}GiB"
    return max_memory

def TaskDispatcher(report_key:str,model_id:str,lora_config:LoraConfig,task:TaskConfig,assigned_gpus:list[int]):
    # TODO: May drop assigned_gpus?
    if not assigned_gpus:
        raise ValueError("TaskDispatcher requires at least one assigned GPU")
    print("Assigned GPUs:",os.environ["CUDA_VISIBLE_DEVICES"],assigned_gpus)

    data = task.dataset
    data = data.map(task.data_formatter, remove_columns=data.column_names)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "left"
    raw_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="balanced",
        # max_memory=build_max_memory(assigned_gpus),
        attn_implementation="flash_attention_2",
        dtype=torch.bfloat16,
    )
    model = get_peft_model(raw_model, lora_config)
    trainer = task.trainer_class(
        model=model,
        args=task.trainer_config,
        train_dataset=data,
        **task.trainer_kwargs
    )
    wandb.login(report_key)
    wandb.init(project=task.name,entity=task.entity)
    trainer.train()
