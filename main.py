from transformers import BertTokenizer, BertForSequenceClassification
import torch

model = BertForSequenceClassification.from_pretrained("./finetuned-bert-news-classifier")
tokenizer = BertTokenizer.from_pretrained("./finetuned-bert-news-classifier")

text = "Apple releases the new iPhone 12 with amazing features."

inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

outputs = model(**inputs)
predictions = torch.argmax(outputs.logits, dim=-1)

categories = ['World', 'Sports', 'Business', 'Sci/Tech']
predicted_category = categories[predictions.item()]

print(f'The predicted category is: {predicted_category}')