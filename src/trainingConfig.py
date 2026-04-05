from typing import Any

from attr import dataclass
from datasets import Dataset
from peft import LoraConfig

@dataclass
class TaskConfig:
    name: str
    dataset: Dataset
    trainer_class: Any
    trainer_config: Any
    trainer_kwargs: dict = None
    reward_func: Any = None

@dataclass
class MultituneConfig:
    model_id: str
    lora_config: LoraConfig
    tasks: list[TaskConfig]