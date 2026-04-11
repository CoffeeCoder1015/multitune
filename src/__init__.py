from .trainingConfig import HFLoraOverrides, LoRAConfigSpec, MultituneConfig, TaskConfig, UnslothLoraOverrides

__all__ = [
    "HFLoraOverrides",
    "LoRAConfigSpec",
    "MultituneConfig",
    "TaskConfig",
    "UnslothLoraOverrides",
    "HFMultitune",
    "UnslothMultitune",
]

def __getattr__(name):
    if name == "UnslothMultitune":
        from .unsloth_dispatch import UnslothMultitune
        return UnslothMultitune

    if name == "HFMultitune":
        from .hf_dispatch import HFMultitune
        return HFMultitune

    raise AttributeError(...)
