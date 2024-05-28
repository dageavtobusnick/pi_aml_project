from transformers import AutoTokenizer, pipeline
from transformers import AutoModelForSequenceClassification

model_name = "typeform/distilbert-base-uncased-mnli"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
classifier = pipeline('zero-shot-classification', model=model, tokenizer=tokenizer)


def predict(texts, labels):
    if not isinstance(texts, list):
        texts = [texts] 
    results = classifier(texts, labels)
    return [(result['labels'][0], result['scores'][0]) for result in results]
