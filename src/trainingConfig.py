from typing import Any, Callable

from attr import dataclass, field
from datasets import Dataset
from peft import LoraConfig

@dataclass
class TaskConfig:
    name: str
    dataset: Dataset
    data_formatter: Callable
    trainer_config: Any
    trainer_class: Any
    trainer_kwargs: dict = field(factory=dict)
    reward_func: Any = None
    entity: str = "messing_around"

@dataclass
class MultituneConfig:
    model_id: str
    lora_config: LoraConfig
    tasks: list[TaskConfig]
