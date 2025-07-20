import argparse
import json
import os
import random
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset, Dataset

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_data(jsonl_path):
    with open(jsonl_path, 'r') as f:
        lines = [json.loads(line) for line in f]
    texts = [x['text'] for x in lines]
    labels = [1 if x['label'] == 'positive' else 0 for x in lines]
    return Dataset.from_dict({'text': texts, 'label': labels})

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=3e-5)
    args = parser.parse_args()

    set_seed(42)

    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    dataset = load_data(args.data)

    def preprocess(examples):
        return tokenizer(examples['text'], truncation=True, padding=True)

    dataset = dataset.map(preprocess, batched=True)

    training_args = TrainingArguments(
        output_dir="./model",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=8,
        learning_rate=args.lr,
        save_strategy="epoch",
        logging_steps=10,
        evaluation_strategy="no",
        save_total_limit=1,
        remove_unused_columns=False,
        fp16=False,
        seed=42
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model("./model")
    tokenizer.save_pretrained("./model")
    print("Fine-tuning complete. Weights saved to ./model/")

if __name__ == "__main__":
    main()
