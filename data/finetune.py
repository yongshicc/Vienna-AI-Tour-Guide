import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch

# 1. Load the JSONL dataset
dataset = load_dataset(
    "json",
    data_files="data/persona_dialogues.jsonl",
    split="train"
)

# 2. Format examples: concatenate user query and bot response
def format_example(example):
    prompt = f"{example['persona']}: User: {example['user']}
Bot:"
    response = example['bot']
    return {"text": prompt + " " + response}

dataset = dataset.map(format_example)

# 3. Load base model and tokenizer
base_model = os.getenv("BASE_MODEL", "Nous-Hermes-2-Mistral-7B-DPO-GGUF")
device_map = "auto"
# Load in 8-bit for efficiency
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    load_in_8bit=True,
    device_map=device_map,
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(base_model)
# Ensure pad token is set
tokenizer.pad_token = tokenizer.eos_token

# 4. Prepare model for k-bit training
model = prepare_model_for_kbit_training(model)

# 5. Add LoRA adapters
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none"
)
model = get_peft_model(model, peft_config)

# 6. Tokenize the data
def tokenize_function(example):
    tokens = tokenizer(
        example["text"],
        truncation=True,
        max_length=512,
        padding="max_length"
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# 7. Set up training arguments
training_args = TrainingArguments(
    output_dir="./checkpoints",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    learning_rate=1e-4,
    logging_steps=10,
    save_steps=100,
    fp16=True,
    optim="paged_adamw_32bit",
    push_to_hub=False
)

# 8. Initialize Trainer and train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer
)

trainer.train()

# 9. Save LoRA adapters and tokenizer
model.save_pretrained("./checkpoints")
tokenizer.save_pretrained("./checkpoints")
