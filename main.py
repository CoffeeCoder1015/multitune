from src import Multitune, MultituneConfig, TaskConfig
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer


# Datasets
medical_reasoning = load_dataset("FreedomIntelligence/medical-o1-reasoning-SFT","en", split = "train[:10000]")
chat_to_sql = load_dataset("philschmid/gretel-synthetic-text-to-sql")

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

def chat_sql_formatter(example):
    system_message = "You are a text to SQL query translator. Users will ask you questions in English and you will generate a SQL query based on the provided SCHEMA."
    user_prompt = f"### SCHEMA:\n{example['sql_context']}\n\n### USER QUERY:\n{example['sql_prompt']}"
    response = f"\n\n### SQL QUERY:\n{example['sql']}"

    prompt = [
        {"role":"system","content":system_message},
        {"role":"user","content":user_prompt}
    ]
    completion = [
        { "role":"assistant", "content":response }
    ]

    return {
        "prompt": prompt,
        "completion": completion
    }


# Training configuration
config = MultituneConfig(
    model_id="unsloth/Qwen3-4B-bnb-4bit",
    lora_config={
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
    },
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
                per_device_train_batch_size=8,
                gradient_accumulation_steps=4,
                learning_rate=2e-4,
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
        # TaskConfig(
        #     name="chat_to_sql",
        #     data_formatter=chat_sql_formatter,
        #     dataset=chat_to_sql,
        #     trainer_class=SFTTrainer,
        #     trainer_config=SFTConfig(
        #         # Output and reporting
        #         output_dir="output/chat_to_sql",
        #         report_to=["wandb"],

        #         # Precision
        #         bf16=True,

        #         # Optimization
        #         per_device_train_batch_size=2,
        #         gradient_accumulation_steps=4,
        #         learning_rate=2e-5,
        #         num_train_epochs=1,

        #         # Logging
        #         logging_steps=10,

        #         # Checkpointing
        #         save_strategy="steps",
        #         save_steps=500,

        #         # Packing
        #         packing=True,

        #         # Seed
        #         seed=900913,
        #     ),
        # )
    ],
)

if __name__ == "__main__":
    multitune = Multitune(config)
    multitune.finetune()
