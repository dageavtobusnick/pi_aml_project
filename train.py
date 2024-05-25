import torch
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
from datasets import load_dataset, load_metric
from sklearn.model_selection import train_test_split

dataset = load_dataset("ag_news")

train_dataset, val_dataset = train_test_split(dataset['train'],
                                              test_size=0.2, random_state=42)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained(
                    "bert-base-uncased", num_labels=4)


def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length",
                     truncation=True, max_length=512)


train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

train_dataset = train_dataset.rename_column("label", "labels")
val_dataset = val_dataset.rename_column("label", "labels")

train_dataset.set_format("torch",
                         columns=["input_ids", "attention_mask", "labels"])
val_dataset.set_format("torch",
                       columns=["input_ids", "attention_mask", "labels"])

metric = load_metric("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(logits, dim=-1)
    return metric.compute(predictions=predictions, references=labels)


training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
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

model.save_pretrained("./finetuned-bert-news-classifier")
tokenizer.save_pretrained("./finetuned-bert-news-classifier")
