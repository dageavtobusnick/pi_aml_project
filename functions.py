from transformers import BertTokenizer


def tokenize_function(text):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    tokenizer.save_pretrained("./finetuned-bert-news-classifier")
    return tokenizer(text, padding="max_length", truncation=True)