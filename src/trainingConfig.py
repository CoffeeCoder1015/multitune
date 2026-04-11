from __future__ import annotations

from typing import Any, Callable

from attr import dataclass, field
from datasets import Dataset


@dataclass
class HFLoraOverrides:
    exclude_modules: list[str] | str | None = None
    fan_in_fan_out: bool = False
    rank_pattern: dict = {}
    alpha_pattern: dict = {}
    megatron_config: dict | None = None
    megatron_core: str = "megatron.core"
    trainable_token_indices: list[int] | dict[str, list[int]] | None = None
    eva_config: Any = None
    corda_config: Any = None
    use_dora: bool = False
    alora_invocation_tokens: list[int] | None = None
    layer_replication: list[tuple[int, int]] | None = None
    runtime_config: Any = None
    lora_bias: bool = False
    task_type: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {key: value for key, value in self.__dict__.items() if value is not None}


@dataclass
class UnslothLoraOverrides:
    use_gradient_checkpointing: str | bool = "unsloth"
    random_state: int = 3407
    max_seq_length: int = 2048
    temporary_location: str = "_unsloth_temporary_saved_buffers"
    qat_scheme: Any = None
    ensure_weight_tying: bool = False
    kwargs: dict[str, Any] = field(factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = {key: value for key, value in self.__dict__.items() if key != "kwargs" and value is not None}
        data.update(self.kwargs)
        return data


@dataclass
class LoRAConfigSpec:
    r: int
    target_modules: list[str]
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    bias: str = "none"
    layers_to_transform: list[int] | int | None = None
    layers_pattern: list[str] | str | None = None
    use_rslora: bool = False
    modules_to_save: list[str] | None = None
    init_lora_weights: bool | str = True
    loftq_config: dict | None = None
    target_parameters: list[str] | None = None
    hf_overrides: HFLoraOverrides = field(factory=HFLoraOverrides)
    unsloth_overrides: UnslothLoraOverrides = field(factory=UnslothLoraOverrides)

    def _base_dict(self) -> dict[str, Any]:
        data = {
            "r": self.r,
            "target_modules": self.target_modules,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "bias": self.bias,
            "layers_to_transform": self.layers_to_transform,
            "layers_pattern": self.layers_pattern,
            "use_rslora": self.use_rslora,
            "modules_to_save": self.modules_to_save,
            "init_lora_weights": self.init_lora_weights,
            "loftq_config": self.loftq_config,
            "target_parameters": self.target_parameters,
        }
        return {key: value for key, value in data.items() if value is not None}

    def to_hf_dict(self) -> dict[str, Any]:
        data = self._base_dict()
        data.update(self.hf_overrides.to_dict())
        return data

    def to_unsloth_dict(self) -> dict[str, Any]:
        data = self._base_dict()
        data.update(self.unsloth_overrides.to_dict())
        return data

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
    lora_config: LoRAConfigSpec
    tasks: list[TaskConfig]
