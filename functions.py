from transformers import BertTokenizer, BertForSequenceClassification
import torch


def tokenize_function(text):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    tokenizer.save_pretrained("./finetuned-bert-news-classifier")
    return tokenizer(text['text'], padding="max_length", truncation=True)


def predict(text):
    model = BertForSequenceClassification.from_pretrained(
                        "./finetuned-bert-news-classifier")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    inputs = tokenizer(text, return_tensors="pt",
                       padding=True, truncation=True, max_length=512)
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1)
    categories = ['World', 'Sports', 'Business', 'Sci/Tech']
    return categories[predictions.item()]
