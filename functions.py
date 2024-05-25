from transformers import BertTokenizer, BertForSequenceClassification
import torch


def tokenize_function(text):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    tokenizer.save_pretrained("./finetuned-bert-news-classifier")
    return tokenizer(text, padding="max_length", truncation=True)


def predict(text):
    model = BertForSequenceClassification.from_pretrained(
                        "./finetuned-bert-news-classifier")
    inputs = tokenize_function(text)
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1)
    categories = ['World', 'Sports', 'Business', 'Sci/Tech']
    return categories[predictions.item()]
