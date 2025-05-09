import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import get_peft_model, LoraConfig, TaskType
import torch

from huggingface_hub import login
login("token")

# === CONFIGURAÇÕES ===
#MODEL_NAME = "mistralai/Mistral-7B-v0.1"
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DATASET_PATH = "data/pmbok_instruct_dataset.jsonl"
OUTPUT_DIR = "./pmbok-model"

# === CARREGAR MODELO E TOKENIZER (SEM CUDA, SEM quantização) ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

tokenizer.pad_token = tokenizer.eos_token

# === Forçar uso da CPU ===
device = torch.device("cpu")
model.to(device)

# === APLICAR LORA (opcional, para treino leve) ===
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)

# === CARREGAR DATASET ===
dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

# === TOKENIZAÇÃO ===
def tokenize(example):
    prompt = f"Instrução: {example['instruction']}\n\nResposta: {example['output']}"
    return tokenizer(prompt, truncation=True, padding="max_length", max_length=512)

tokenized_dataset = dataset.map(tokenize)

# === ARGUMENTOS DE TREINAMENTO ===
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    num_train_epochs=1,  # Você pode aumentar depois
    logging_steps=10,
    save_strategy="epoch",
    logging_dir="./logs",
)

# === INICIALIZAR TREINADOR ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

# === INICIAR TREINAMENTO ===
trainer.train()
