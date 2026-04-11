from src import HFMultitune, MultituneConfig, TaskConfig
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer


# Datasets
medical_reasoning = load_dataset("FreedomIntelligence/medical-o1-reasoning-SFT","en", split = "train[:10000]")
chat_to_sql = load_dataset("philschmid/gretel-synthetic-text-to-sql",split="train")

# Formatting
def medical_formatter(example):
    # Define the system and instruction texts
    system_prompt = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response."
    
    instruction = "You are a medical expert with advanced knowledge in clinical reasoning, diagnostics, and treatment planning. Please answer the following medical question."
    
    # Merge everything into a single formatted string using the tutorial's structure
    text = f"### System:\n{system_prompt}\n\n### Instruction:\n{instruction}\n\n### Question:\n{example['Question']}\n\n### Chain of Thought:\n{example['Complex_CoT']}\n\n### Response:\n{example['Response']}"
    
    # Return a dictionary with the single "text" key
    return {"text": text}


def chat_sql_formatter(example):
    # Define the system message
    system_message = "You are a text to SQL query translator. Users will ask you questions in English and you will generate a SQL query based on the provided SCHEMA."

    # Merge everything into a single formatted string using the tutorial's structure
    text = f"### System:\n{system_message}\n\n### Schema:\n{example['sql_context']}\n\n### User Query:\n{example['sql_prompt']}\n\n### SQL Query:\n{example['sql']}"

    # Return a dictionary with the single "text" key
    return {"text": text}

# LoRA configurations
unsloth_lora_config = {
    "lora_alpha": 16,
    "r": 64,
    "lora_dropout": 0,
    "bias": "none",
    "use_gradient_checkpointing": "unsloth",
    "random_state": 3407,
    "target_modules": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
}

hf_lora_config = LoraConfig(
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
)


# Training configuration
config = MultituneConfig(
    model_id="unsloth/Qwen3-32B",
    lora_config=unsloth_lora_config,
    tasks=[
        TaskConfig(
            name="medical_reasoning",
            data_formatter=medical_formatter,
            dataset=medical_reasoning,
            trainer_class=SFTTrainer,
            trainer_config=SFTConfig(
                # Output and reporting
                output_dir="output/medical_reasoning",
                report_to=["wandb"],

                # Optimization
                per_device_train_batch_size=16,
                gradient_accumulation_steps=4,
                learning_rate=2e-3,
                num_train_epochs=3,

                # Logging
                logging_steps=10,

                # Checkpointing
                # save_safetensors=True,      # recommended
                save_strategy="steps",
                save_steps=500,

                # Packing
                packing=True,
            ),
        ),
        TaskConfig(
            name="chat_to_sql",
            data_formatter=chat_sql_formatter,
            dataset=chat_to_sql,
            trainer_class=SFTTrainer,
            trainer_config=SFTConfig(
                # Output and reporting
                output_dir="output/chat_to_sql",
                report_to=["wandb"],

                # Precision
                bf16=True,

                # Optimization
                per_device_train_batch_size=16,
                gradient_accumulation_steps=4,
                learning_rate=2e-5,
                num_train_epochs=1,

                # Logging
                logging_steps=10,

                # Checkpointing
                save_strategy="steps",
                save_steps=500,

                # Packing
                packing=True,

                # Seed
                seed=900913,
            ),
        )
    ],
)

if __name__ == "__main__":
    multitune = HFMultitune(config)
    multitune.finetune()
