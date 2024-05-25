from transformers import BertForSequenceClassification
import torch
import functions

model = BertForSequenceClassification.from_pretrained(
                    "./finetuned-bert-news-classifier")

text = "Apple releases the new iPhone 12 with amazing features."

inputs = functions.tokenize_function(text)

outputs = model(**inputs)
predictions = torch.argmax(outputs.logits, dim=-1)

categories = ['World', 'Sports', 'Business', 'Sci/Tech']
predicted_category = categories[predictions.item()]

print(f'The predicted category is: {predicted_category}')
