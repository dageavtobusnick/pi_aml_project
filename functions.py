from transformers import AutoTokenizer, pipeline
from transformers import AutoModelForSequenceClassification

model_name = "typeform/distilbert-base-uncased-mnli"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
classifier = pipeline('zero-shot-classification',
                      model=model,
                      tokenizer=tokenizer)


def predict(texts, labels):
    if not isinstance(texts, (list, str)):
        raise ValueError("The 'texts' parameter should be" +
                         " a string or a list of strings.")
    if isinstance(texts, str):
        texts = [texts]

    if not isinstance(labels, list) or not all(isinstance(label, str)
                                               for label in labels):
        raise ValueError("The 'labels' parameter should be a list of strings.")

    if not texts:
        raise ValueError("The 'texts' list should not be empty.")
    if not labels:
        raise ValueError("The 'labels' list should not be empty.")

    results = classifier(texts, labels)
    return [(result['labels'][0], result['scores'][0]) for result in results]
