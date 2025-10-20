# train.py - LoRA fine-tuning for AGI safety
import os
import json
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset

def main():
    # Parse hyperparameters
    model_name = os.environ.get('SM_HP_MODEL_NAME_OR_PATH', 'deepseek-ai/deepseek-coder-6.7b-base')
    epochs = int(os.environ.get('SM_HP_EPOCHS', 2))
    batch_size = int(os.environ.get('SM_HP_BATCH_SIZE', 4))
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # LoRA configuration for AGI safety
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
    )
    
    model = get_peft_model(model, lora_config)
    
    # Load your mycelia recursion data
    train_path = os.environ['SM_CHANNEL_TRAIN']
    with open(f"{train_path}/mycelia-data.jsonl", 'r') as f:
        data = [json.loads(line) for line in f]
    
    # Tokenize dataset
    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True, padding=True, max_length=512)
    
    dataset = Dataset.from_list(data)
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=os.environ['SM_MODEL_DIR'],
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="no",
        fp16=True,
        dataloader_pin_memory=False,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    
    # Train the model
    trainer.train()
    trainer.save_model()

if __name__ == "__main__":
    main()
