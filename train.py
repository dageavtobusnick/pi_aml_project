import torch
from transformers import BertForSequenceClassification
from transformers import Trainer, TrainingArguments
from datasets import load_dataset, load_metric, DatasetDict
from sklearn.model_selection import train_test_split
import functions


def compute_metrics(eval_pred):
    metric = load_metric("accuracy")
    logits, labels = eval_pred
    predictions = torch.argmax(logits, dim=-1)
    return metric.compute(predictions=predictions, references=labels)


def load_and_split_dataset(dataset_name, split_ratio=0.2, seed=42):
    dataset = load_dataset(dataset_name)
    train_dataset, val_dataset = train_test_split(dataset['train'],
                                                  test_size=split_ratio,
                                                  random_state=seed)
    return DatasetDict({"train": train_dataset, "validation": val_dataset})


def tokenize_and_format_datasets(datasets, max_length=512):
    tokenized_datasets = datasets.map(functions.tokenize_function,
                                      batched=True)
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch",
                                  columns=["input_ids",
                                           "attention_mask",
                                           "labels"])
    return tokenized_datasets


def train_model(train_dataset, val_dataset, model_name, output_dir,
                epochs=3, batch_size=8, learning_rate=2e-5):
    model = BertForSequenceClassification.from_pretrained(
        model_name, num_labels=4)

    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        logging_dir=f'{output_dir}/logs',
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    model.save_pretrained(output_dir)
    return model


dataset_name = "ag_news"
model_name = "bert-base-uncased"
output_dir = "./finetuned-bert-news-classifier"

datasets = load_and_split_dataset(dataset_name)
tokenized_datasets = tokenize_and_format_datasets(datasets)
model = train_model(tokenized_datasets['train'],
                    tokenized_datasets['validation'],
                    model_name, output_dir)
