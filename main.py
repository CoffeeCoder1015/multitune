from src import LoRAConfigSpec, MultituneConfig, TaskConfig, UnslothLoraOverrides, UnslothMultitune
from datasets import load_dataset
from src.trainingConfig import HFLoraOverrides
import random
from trl import SFTConfig, SFTTrainer


# Datasets
snli = load_dataset("snli", split="train[:50_000]")
snli_val = load_dataset("snli", split="validation")
logic_fallacy = load_dataset("tasksource/logical-fallacy",split="train")
logic_fallacy_val = load_dataset("tasksource/logical-fallacy",split="dev")

FALLACY_PROMPT_VARIATIONS = [
    lambda p : f"What logical fallacy is: {p}",
    lambda p : f"Examine \"{p}\" and identify what logical fallacy is it. ",
    lambda p : f"{p}, the logical fallacy is:",
    lambda p : f"Analyze the following statement for logical inconsistencies: \"{p}\". Provide the name of the fallacy and a brief justification.",
    lambda p : f"Instruction: Categorize the logical error in the text below.\nText: {p}\nFallacy Category:"
]

def fallacy_formatter(example):
    prompt_fn = random.choice(FALLACY_PROMPT_VARIATIONS)
    base_prompt = example["source_article"]
    answer = example["logical_fallacies"]

    prompt = [ {"role":"user","content":prompt_fn(base_prompt)} ]
    completion = [ {"role":"assistant","content":example["logical_fallacies"]} ]
    return {
        "prompt":prompt,
        "completion":completion
    }

NLI_PROMPT_VARIATIONS = [
    lambda p, h: f"Is the hypothesis entailed, neutral, or contradictory to the premise? Premise: {p} Hypothesis: {h}",
    lambda p, h: f"What is the relationship between premise and hypothesis? Premise: {p} Hypothesis: {h}",
    lambda p, h: f"Inference the relationship between the Premise: {p} and Hypothesis: {h}",
    lambda p, h: f"Classify as entailment, neutral, or contradiction.\nPremise: {p}\nHypothesis: {h}",
    lambda p, h: f"Premise: {p}\nHypothesis: {h}\nRelationship:",
]

CLASSIFICATION_MAP = ["entailment", "neutral", "contradiction"]

def snli_formatter(example):
    prompt_fn = random.choice(NLI_PROMPT_VARIATIONS)
    prompt = [ {"role":"user","content":prompt_fn(example["premise"], example["hypothesis"])} ]

    label = CLASSIFICATION_MAP[example["label"]]
    completion = [ {"role":"assistant", "content":label}]

    example["prompt"] = prompt
    example["completion"] = completion
    return example


# LoRA configuration
lora_config = LoRAConfigSpec(
    lora_alpha=128,
    r=64,
    lora_dropout=0,
    bias="none",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    hf_overrides=HFLoraOverrides(
        task_type="CAUSAL_LM"
    )
)


# Training configuration
config = MultituneConfig(
    model_id="LiquidAI/LFM2.5-1.2B-Thinking",
    lora_config=lora_config,
    tasks=[
        TaskConfig(
            name="snli",
            dataset=snli,
            data_formatter=snli_formatter,
            trainer_class=SFTTrainer,
            trainer_config=SFTConfig(
                # Output and reporting
                output_dir="output/snli",
                report_to=["wandb"],

                # Optimization
                per_device_train_batch_size=8,
                gradient_accumulation_steps=2,
                learning_rate=2e-4,
                num_train_epochs=5,

                # Logging
                logging_steps=10,

                # Checkpointing
                save_strategy="steps",
                save_steps=250,
                save_total_limit=5,

                # Packing
                packing=True,
                
                # Eval
                eval_strategy="steps",
                eval_steps=50,
            ),
            trainer_kwargs={
                "eval_dataset":snli_val.map(snli_formatter)
            }
        ),
        TaskConfig(
            name="fallacy",
            dataset=logic_fallacy,
            data_formatter=fallacy_formatter,
            trainer_class=SFTTrainer,
            trainer_config=SFTConfig(
                # Output and reporting
                output_dir="output/fallacy",
                report_to=["wandb"],

                # Optimization
                per_device_train_batch_size=8,
                gradient_accumulation_steps=2,
                learning_rate=2e-4,
                num_train_epochs=5,

                # Logging
                logging_steps=10,

                # Checkpointing
                save_strategy="steps",
                save_steps=250,
                save_total_limit=5,

                # Packing
                packing=True,
                
                # Eval
                eval_strategy="steps",
                eval_steps=50,
            ),
            trainer_kwargs={
                "eval_dataset":logic_fallacy_val.map(fallacy_formatter)
            }
        )
    ],
)

if __name__ == "__main__":
    multitune = UnslothMultitune(config)
    multitune.finetune()
