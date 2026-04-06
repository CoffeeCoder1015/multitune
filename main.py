from src import Multitune, MultituneConfig, TaskConfig
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer


# Dataset
medical_reasoning = load_dataset("FreedomIntelligence/medical-o1-reasoning-SFT","en", split = "train[:10000]")


# Formatting
def medical_formatter(example):
    return {
        "prompt": [{"role": "user", "content": example["Question"]}],
        "completion": [
            {
                "role": "assistant",
                "content": f"<think>{example['Complex_CoT']}</think>{example['Response']}",
            }
        ],
    }


# Training configuration
config = MultituneConfig(
    model_id="Qwen/Qwen3.5-4B-Base",
    lora_config=LoraConfig(
        lora_alpha=16,                      
        r=64,
        bias="none", 
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],  
    ),
    tasks=[
        TaskConfig(
            name="medical_reasoning",
            data_formatter=medical_formatter,
            dataset=medical_reasoning,
            trainer_class=SFTTrainer,
            trainer_config=SFTConfig(
                # Output and reporting
                output_dir="medical_reasoning",
                report_to=["wandb"],

                # Optimization
                per_device_train_batch_size=8,
                gradient_accumulation_steps=4,
                learning_rate=2e-4,
                num_train_epochs=3,

                # Logging
                logging_steps=10,

                # Checkpointing
                save_safetensors=True,      # recommended
                save_strategy="steps",
                save_steps=500,

                # Packing
                packing=True,
            ),
        )
    ],
)

if __name__ == "__main__":
    multitune = Multitune(config)
    multitune.finetune()
